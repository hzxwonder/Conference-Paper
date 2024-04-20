## [800] MAIR: Multi-View Attention Inverse Rendering with 3D Spatially-Varying Lighting Estimation

        **Authors**: *Junyong Choi, SeokYeong Lee, Haesol Park, Seung-Won Jung, Ig-Jae Kim, Junghyun Cho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00811](https://doi.org/10.1109/CVPR52729.2023.00811)

        **Abstract**:

        We propose a scene-level inverse rendering framework that uses multi-view images to decompose the scene into geometry, a SVBRDF, and 3D spatially-varying lighting. Because multi-view images provide a variety of information about the scene, multi-view images in object-level inverse rendering have been taken for granted. However, owing to the absence of multi-view HDR synthetic dataset, scene-level inverse rendering has mainly been studied using single-view image. We were able to successfully perform scene-level inverse rendering using multi-view images by expanding OpenRooms dataset and designing efficient pipelines to handle multi-view images, and splitting spatially-varying lighting. Our experiments show that the proposed method not only achieves better performance than single-view-based methods, but also achieves robust performance on unseen real-world scene. Also, our sophisticated 3D spatially-varying lighting volume allows for photorealistic object insertion in any 3D location.

        ----

        ## [801] Weakly-supervised Single-view Image Relighting

        **Authors**: *Renjiao Yi, Chenyang Zhu, Kai Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00812](https://doi.org/10.1109/CVPR52729.2023.00812)

        **Abstract**:

        We present a learning-based approach to relight a single image of Lambertian and low-frequency specular objects. Our method enables inserting objects from photographs into new scenes and relighting them under the new environment lighting, which is essential for AR applications. To relight the object, we solve both inverse rendering and re-rendering. To resolve the ill-posed inverse rendering, we propose a weakly-supervised method by a low-rank constraint. To facilitate the weakly-supervised training, we contribute Relit, a large-scale (750K images) dataset of videos with aligned objects under changing illuminations. For re-rendering, we propose a differentiable specular rendering layer to render low-frequency non-Lambertian materials under various illuminations of spherical harmonics. The whole pipeline is end-to-end and efficient, allowing for a mobile app implementation of AR object insertion. Extensive evaluations demonstrate that our method achieves state-of-the-art performance. Project page: https://renjiaoyi.github.io/relighting/.

        ----

        ## [802] Controllable Light Diffusion for Portraits

        **Authors**: *David Futschik, Kelvin Ritland, James Vecore, Sean Fanello, Sergio Orts-Escolano, Brian Curless, Daniel Sýkora, Rohit Pandey*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00813](https://doi.org/10.1109/CVPR52729.2023.00813)

        **Abstract**:

        We introduce light diffusion, a novel method to improve lighting in portraits, softening harsh shadows and specular highlights while preserving overall scene illumi-nation. Inspired by professional photographers' diffusers and scrims, our method softens lighting given only a single portrait photo. Previous portrait relighting approaches focus on changing the entire lighting environment, removing shadows (ignoring strong specular highlights), or removing shading entirely. In contrast, we propose a learning based method that allows us to control the amount of light diffusion and apply it on in-the-wild portraits. Additionally, we design a method to synthetically generate plausible external shadows with sub-surface scattering effects while conforming to the shape of the subject's face. Finally, we show how our approach can increase the robustness of higher level vision applications, such as albedo estimation, geometry estimation and semantic segmentation.

        ----

        ## [803] RGBD2: Generative Scene Synthesis via Incremental View Inpainting Using RGBD Diffusion Models

        **Authors**: *Jiabao Lei, Jiapeng Tang, Kui Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00814](https://doi.org/10.1109/CVPR52729.2023.00814)

        **Abstract**:

        We address the challenge of recovering an underlying scene geometry and colors from a sparse set of RGBD view observations. In this work, we present a new solution termed RGBD
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 that sequentially generates novel RGBD views along a camera trajectory, and the scene geometry is simply the fusion result of these views. More specifically, we maintain an intermediate surface mesh used for rendering new RGBD views, which subsequently becomes complete by an inpainting network; each rendered RGBD view is later back-projected as a partial surface and is supplemented into the intermediate mesh. The use of intermediate mesh and camera projection helps solve the tough problem of multi-view inconsistency. We practically implement the RGBD inpainting network as a versatile RGBD diffusion model, which is previously used for 2D generative modeling; we make a modification to its reverse diffusion process to enable our use. We evaluate our approach on the task of 3D scene synthesis from sparse RGBD inputs; extensive experiments on the ScanNet dataset demonstrate the superiority of our approach over existing ones. Project page: https://jblei.site/proj/rgbd-diffusion.

        ----

        ## [804] Neural Lens Modeling

        **Authors**: *Wenqi Xian, Aljaz Bozic, Noah Snavely, Christoph Lassner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00815](https://doi.org/10.1109/CVPR52729.2023.00815)

        **Abstract**:

        Recent methods for 3D reconstruction and rendering increasingly benefit from end-to-end optimization of the entire image formation process. However, this approach is currently limited: effects of the optical hardware stack and in particular lenses are hard to model in a unified way. This limits the quality that can be achieved for camera calibration and the fidelity of the results of 3D reconstruction. In this paper, we propose NeuroLens, a neural lens model for distortion and vignetting that can be used for point projection and ray casting and can be optimized through both operations. This means that it can (optionally) be used to perform pre-capture calibration using classical calibration targets, and can later be used to perform calibration or refinement during 3D reconstruction, e.g., while optimizing a radiance field. To evaluate the performance of our proposed model, we create a comprehensive dataset assembled from the Lensfun database with a multitude of lenses. Using this and other real-world datasets, we show that the quality of our proposed lens model outperforms standard packages as well as recent approaches while being much easier to use and extend. The model generalizes across many lens types and is trivial to integrate into existing 3D reconstruction and rendering systems. Visit our project website at: https://neural-lens.github.io.

        ----

        ## [805] RealFusion 360° Reconstruction of Any Object from a Single Image

        **Authors**: *Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Andrea Vedaldi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00816](https://doi.org/10.1109/CVPR52729.2023.00816)

        **Abstract**:

        We consider the problem of reconstructing a full 360° photographic model of an object from a single image of it. We do so by fitting a neural radiance field to the image, but find this problem to be severely ill-posed. We thus take an off-the-self conditional image generator based on diffusion and engineer a prompt that encourages it to “dream up” novel views of the object. Using the recent DreamFusion method, we fuse the given input view, the conditional prior, and other regularizers into a final, consistent reconstruction. We demonstrate state-of-the-art reconstruction results on benchmark images when compared to prior methods for monocular 3D reconstruction of objects. Qualitatively, our reconstructions provide a faithful match of the input view and a plausible extrapolation of its appearance and 3D shape, including to the side of the object not visible in the image.

        ----

        ## [806] Neuralangelo: High-Fidelity Neural Surface Reconstruction

        **Authors**: *Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, Chen-Hsuan Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00817](https://doi.org/10.1109/CVPR52729.2023.00817)

        **Abstract**:

        Neural surface reconstruction has been shown to be powerful for recovering dense 3D surfaces via image-based neural rendering. However, current methods struggle to recover detailed structures of real-world scenes. To address the issue, we present Neuralangelo, which combines the representation power of multiresolution 3D hash grids with neural surface rendering. Two key ingredients enable our approach: (1) numerical gradients for computing higher-order derivatives as a smoothing operation and (2) coarse-to-fine optimization on the hash grids controlling different levels of details. Even without auxiliary inputs such as depth, Neuralangelo can effectively recover dense 3D surface structures from multiview images with fidelity significantly surpassing previous methods, enabling detailed large-scale scene reconstruction from RGB video captures.

        ----

        ## [807] PermutoSDF: Fast Multi-View Reconstruction with Implicit Surfaces Using Permutohedral Lattices

        **Authors**: *Radu Alexandru Rosu, Sven Behnke*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00818](https://doi.org/10.1109/CVPR52729.2023.00818)

        **Abstract**:

        Neural radiance-density field methods have become increasingly popular for the task of novel-view rendering. Their recent extension to hash-based positional encoding ensures fast training and inference with visually pleasing results. However, density-based methods struggle with recovering accurate surface geometry. Hybrid methods alleviate this issue by optimizing the density based on an underlying SDF. However, current SDF methods are overly smooth and miss fine geometric details. In this work, we combine the strengths of these two lines of work in a novel hash-based implicit surface representation. We propose improvements to the two areas by replacing the voxel hash encoding with a permutohedral lattice which optimizes faster, especially for higher dimensions. We additionally propose a regularization scheme which is crucial for recovering high-frequency geometric detail. We evaluate our method on multiple datasets and show that we can recover geometric detail at the level of pores and wrinkles while using only RGB images for supervision. Furthermore, using sphere tracing we can render novel views at 30 fps on an RTX 3090. Code is publicly available at https://radualexandru.github.io/permuto_sdf.

        ----

        ## [808] NeuDA: Neural Deformable Anchor for High-Fidelity Implicit Surface Reconstruction

        **Authors**: *Bowen Cai, Jinchi Huang, Rongfei Jia, Chengfei Lv, Huan Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00819](https://doi.org/10.1109/CVPR52729.2023.00819)

        **Abstract**:

        This paper studies implicit surface reconstruction leveraging differentiable ray casting. Previous works such as IDR [34] and NeuS [27] overlook the spatial context in 3D space when predicting and rendering the surface, thereby may fail to capture sharp local topologies such as small holes and structures. To mitigate the limitation, we propose a flexible neural implicit representation leveraging hierarchical voxel grids, namely Neural Deformable Anchor (NeuDA), for high-fidelity surface reconstruction. NeuDA maintains the hierarchical anchor grids where each vertex stores a 3D position (or anchor) instead of the direct embedding (or feature). We optimize the anchor grids such that different local geometry structures can be adaptively encoded. Besides, we dig into the frequency encoding strategies and introduce a simple hierarchical positional encoding method for the hierarchical anchor structure to flexibly exploit the properties of high-frequency and low-frequency geometry and appearance. Experiments on both the DTU [8] and BlendedMVS [32] datasets demonstrate that NeuDA can produce promising mesh surfaces.

        ----

        ## [809] NEF: Neural Edge Fields for 3D Parametric Curve Reconstruction from Multi-View Images

        **Authors**: *Yunfan Ye, Renjiao Yi, Zhirui Gao, Chenyang Zhu, Zhiping Cai, Kai Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00820](https://doi.org/10.1109/CVPR52729.2023.00820)

        **Abstract**:

        We study the problem of reconstructing 3D feature curves of an object from a set of calibrated multi-view images. To do so, we learn a neural implicit field representing the density distribution of 3D edges which we refer to as Neural Edge Field (NEF). Inspired by NeRF [20], NEF is optimized with a view-based rendering loss where a 2D edge map is rendered at a given view and is compared to the ground-truth edge map extracted from the image of that view. The rendering-based differentiable optimization of NEF fully exploits 2D edge detection, without needing a supervision of 3D edges, a 3D geometric operator or cross-view edge correspondence. Several technical designs are devised to ensure learning a range-limited and view-independent NEF for robust edge extraction. The final parametric 3D curves are extracted from NEF with an iterative optimization method. On our benchmark with synthetic data, we demonstrate that NEF outperforms existing state-of-the-art methods on all metrics. Project page: https://yunfan1202.github.io/NEF/.

        ----

        ## [810] NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models

        **Authors**: *Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, Sanja Fidler*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00821](https://doi.org/10.1109/CVPR52729.2023.00821)

        **Abstract**:

        Automatically generating high-quality real world 3D scenes is of enormous interest for applications such as virtual reality and robotics simulation. Towards this goal, we introduce NeuralField-LDM, a generative model capable of synthesizing complex 3D environments. We leverage Latent Diffusion Models that have been successfully utilized for efficient high-quality 2D content creation. We first train a scene auto-encoder to express a set of image and pose pairs as a neural field, represented as density and feature voxel grids that can be projected to produce novel views of the scene. To further compress this representation, we train a latent-autoencoder that maps the voxel grids to a set of latent representations. A hierarchical diffusion model is then fit to the latents to complete the scene generation pipeline. We achieve a substantial improvement over existing state-of-the-art scene generation models. Additionally, we show how NeuralField-LDM can be used for a variety of 3D content creation applications, including conditional scene generation, scene inpainting and scene style manipulation.

        ----

        ## [811] SinGRAF: Learning a 3D Generative Radiance Field for a Single Scene

        **Authors**: *Minjung Son, Jeong Joon Park, Leonidas J. Guibas, Gordon Wetzstein*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00822](https://doi.org/10.1109/CVPR52729.2023.00822)

        **Abstract**:

        Generative models have shown great promise in synthesizing photorealistic 3D objects, but they require large amounts of training data. We introduce SinGRAF, a 3D-aware generative model that is trained with a few input images of a single scene. Once trained, SinGRAF generates different realizations of this 3D scene that preserve the appearance of the input while varying scene layout. For this purpose, we build on recent progress in 3D GAN architectures and introduce a novel progressive-scale patch discrimination approach during training. With several experiments, we demonstrate that the results produced by Sin-GRAF outperform the closest related works in both quality and diversity by a large margin.

        ----

        ## [812] Painting 3D Nature in 2D: View Synthesis of Natural Scenes from a Single Semantic Mask

        **Authors**: *Shangzhan Zhang, Sida Peng, Tianrun Chen, Linzhan Mou, Haotong Lin, Kaicheng Yu, Yiyi Liao, Xiaowei Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00823](https://doi.org/10.1109/CVPR52729.2023.00823)

        **Abstract**:

        We introduce a novel approach that takes a single semantic mask as input to synthesize multi-view consistent color images of natural scenes, trained with a collection of single images from the Internet. Prior works on 3D-aware image synthesis either require multi-view supervision or learning category-level prior for specific classes of objects, which are inapplicable to natural scenes. Our key idea to solve this challenge is to use a semantic field as the intermediate representation, which is easier to reconstruct from an input semantic mask and then translated to a radiance field with the assistance of off-the-shelf semantic image synthesis models. Experiments show that our method outperforms baseline methods and produces photorealistic and multi-view consistent videos of a variety of natural scenes. The project website is https://zju3dv.github.io/paintingnature/.

        ----

        ## [813] Quantitative Manipulation of Custom Attributes on 3D-Aware Image Synthesis

        **Authors**: *Hoseok Do, Eunkyung Yoo, Taehyeong Kim, Chul Lee, Jin Young Choi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00824](https://doi.org/10.1109/CVPR52729.2023.00824)

        **Abstract**:

        While 3D-based GAN techniques have been successfully applied to render photo-realistic 3D images with a variety of attributes while preserving view consistency, there has been little research on how to fine-control 3D images without limiting to a specific category of objects of their properties. To fill such research gap, we propose a novel image manipulation model of 3D-based GAN representations for a fine-grained control of specific custom attributes. By extending the latest 3D-based GAN models (e.g., EG3D), our user-friendly quantitative manipulation model enables a fine yet normalized control of 3D manipulation of multi-attribute quantities while achieving view consistency. We validate the effectiveness of our proposed technique both qualitatively and quantitatively through various experiments.

        ----

        ## [814] NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-Shot Real Image Animation

        **Authors**: *Yu Yin, Kamran Ghasedi, HsiangTao Wu, Jiaolong Yang, Xin Tong, Yun Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00825](https://doi.org/10.1109/CVPR52729.2023.00825)

        **Abstract**:

        Nerf-based Generative models have shown impressive capacity in generating high-quality images with consistent 3D geometry. Despite successful synthesis of fake identity images randomly sampled from latent space, adopting these models for generating face images of real subjects is still a challenging task due to its so-called inversion issue. In this paper, we propose a universal method to surgically finetune these NeRF-GAN models in order to achieve high-fidelity animation of real subjects only by a single image. Given the optimized latent code for an out-of-domain real image, we employ 2D loss functions on the rendered image to reduce the identity gap. Furthermore, our method leverages explicit and implicit 3D regularizations using the in-domain neighborhood samples around the optimized latent code to remove geometrical and visual artifacts. Our experiments confirm the effectiveness of our method in realistic, high-fidelity, and 3D consistent animation of real faces on multiple NeRF-GAN models across different datasets.

        ----

        ## [815] PREIM3D: 3D Consistent Precise Image Attribute Editing from a Single Image

        **Authors**: *Jianhui Li, Jianmin Li, Haoji Zhang, Shilong Liu, Zhengyi Wang, Zihao Xiao, Kaiwen Zheng, Jun Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00826](https://doi.org/10.1109/CVPR52729.2023.00826)

        **Abstract**:

        We study the 3D-aware image attribute editing problem in this paper, which has wide applications in practice. Recent methods solved the problem by training a shared encoder to map images into a 3D generator's latent space or by per-image latent code optimization and then edited images in the latent space. Despite their promising results near the input view, they still suffer from the 3D inconsistency of produced images at large camera poses and imprecise image attribute editing, like affecting unspecified attributes during editing. For more efficient image inversion, we train a shared encoder for all images. To alleviate 3D inconsistency at large camera poses, we propose two novel methods, an alternating training scheme and a multi-view identity loss, to maintain 3D consistency and subject identity. As for imprecise image editing, we attribute the problem to the gap between the latent space of real images and that of generated images. We compare the latent space and inversion manifold of GAN models and demonstrate that editing in the inversion manifold can achieve better results in both quantitative and qualitative evaluations. Extensive experiments show that our method produces more 3D consistent images and achieves more precise image editing than previous work. Source code and pretrained models can be found on our project page: https://mybabyyh.github.io/Preim3D/.

        ----

        ## [816] Unsupervised 3D Shape Reconstruction by Part Retrieval and Assembly

        **Authors**: *Xianghao Xu, Paul Guerrero, Matthew Fisher, Siddhartha Chaudhuri, Daniel Ritchie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00827](https://doi.org/10.1109/CVPR52729.2023.00827)

        **Abstract**:

        Representing a 3D shape with a set of primitives can aid perception of structure, improve robotic object manipulation, and enable editing, stylization, and compression of 3D shapes. Existing methods either use simple parametric primitives or learn a generative shape space of parts. Both have limitations: parametric primitives lead to coarse approximations, while learned parts offer too little control over the decomposition. We instead propose to decompose shapes using a library of 3D parts provided by the user, giving full control over the choice of parts. The library can contain parts with high-quality geometry that are suitable for a given category, resulting in meaningful decompositions with clean geometry. The type of decomposition can also be controlled through the choice of parts in the library. Our method works via a unsupervised approach that iteratively retrieves parts from the library and refines their placements. We show that this approach gives higher reconstruction accuracy and more desirable decompositions than existing approaches. Additionally, we show how the decomposition can be controlled through the part library by using different part libraries to reconstruct the same shapes.

        ----

        ## [817] DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion

        **Authors**: *Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, Jiwen Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00828](https://doi.org/10.1109/CVPR52729.2023.00828)

        **Abstract**:

        In this paper, we propose DiffSwap, a diffusion model based framework for high-fidelity and controllable face swapping. Unlike previous work that relies on carefully designed network architectures and loss functions to fuse the information from the source and target faces, we reformulate the face swapping as a conditional inpainting task, performed by a powerful diffusion model guided by the desired face attributes (e.g., identity and landmarks). An important issue that makes it nontrivial to apply diffusion models to face swapping is that we cannot perform the time-consuming multi-step sampling to obtain the generated image during training. To overcome this, we propose a mid-point estimation method to efficiently recover a reasonable diffusion result of the swapped face with only 2 steps, which enables us to introduce identity constraints to improve the face swapping quality. Our framework enjoys several favorable properties more appealing than prior arts: 1) Controllable. Our method is based on conditional masked diffusion on the latent space, where the mask and the conditions can be fully controlled and customized. 2) High-fidelity. The formulation of conditional inpainting can fully exploit the generative ability of diffusion models and can preserve the background of target images with minimal artifacts. 3) Shape-preserving. The controllability of our method enables us to use 3D-aware landmarks as the condition during generation to preserve the shape of the source face. Extensive experiments on both FF++ and FFHQ demonstrate that our method can achieve state-of-the-art face swapping results both qualitatively and quantitatively.

        ----

        ## [818] Fine-Grained Face Swapping Via Regional GAN Inversion

        **Authors**: *Zhian Liu, Maomao Li, Yong Zhang, Cairong Wang, Qi Zhang, Jue Wang, Yongwei Nie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00829](https://doi.org/10.1109/CVPR52729.2023.00829)

        **Abstract**:

        We present a novel paradigm for high-fidelity face swapping that faithfully preserves the desired subtle geometry and texture details. We rethink face swapping from the perspective of fine-grained face editing, i.e., “editing for swapping” (E4S), and propose a framework that is based on the explicit disentanglement of the shape and texture of facial components. Following the E4S principle, our framework enables both global and local swapping of facial features, as well as controlling the amount of partial swapping specified by the user. Furthermore, the E4S paradigm is in-herently capable of handling facial occlusions by means of facial masks. At the core of our system lies a novel Regional GAN Inversion (RGI) method, which allows the explicit disentanglement of shape and texture. It also allows face swapping to be performed in the latent space of Style-GAN. Specifically, we design a multi-scale mask-guided encoder to project the texture of each facial component into regional style codes. We also design a mask-guided injection module to manipulate the feature maps with the style codes. Based on the disentanglement, face swapping is re-formulated as a simplified problem of style and mask swapping. Extensive experiments and comparisons with current state-of-the-art methods demonstrate the superiority of our approach in preserving texture and shape details, as well as working with high resolution images. The project page is https://e4s2022.github.io

        ----

        ## [819] Logical Consistency and Greater Descriptive Power for Facial Hair Attribute Learning

        **Authors**: *Haiyu Wu, Grace Bezold, Aman Bhatta, Kevin W. Bowyer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00830](https://doi.org/10.1109/CVPR52729.2023.00830)

        **Abstract**:

        Face attribute research has so far used only simple binary attributes for facial hair; e.g., beard / no beard. We have created a new, more descriptive facial hair annotation scheme and applied it to create a new facial hair attribute dataset, FH37K. Face attribute research also so far has not dealt with logical consistency and completeness. For example, in prior research, an image might be classified as both having no beard and also having a goatee (a type of beard). We show that the test accuracy of previous classification methods on facial hair attribute classification drops significantly if logical consistency of classifications is enforced. We propose a logically consistent prediction loss, LCPLoss, to aid learning of logical consistency across attributes, and also a label compensation training strategy to eliminate the problem of no positive prediction across a set of related attributes. Using an attribute classifier trained on FH37K, we investigate how facial hair affects face recognition accuracy, including variation across demographics. Results show that similarity and difference in facial hairstyle have important effects on the impostor and genuine score distributions in face recognition. The code is at https://github.com/HaiyuWu/LogicalConsistency.

        ----

        ## [820] Learning a 3D Morphable Face Reflectance Model from Low-Cost Data

        **Authors**: *Yuxuan Han, Zhibo Wang, Feng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00831](https://doi.org/10.1109/CVPR52729.2023.00831)

        **Abstract**:

        Modeling non-Lambertian effects such as facial specularity leads to a more realistic 3D Morphable Face Model. Existing works build parametric models for diffuse and specular albedo using Light Stage data. However, only diffuse and specular albedo cannot determine the full BRDF. In addition, the requirement of Light Stage data is hard to fulfill for the research communities. This paper proposes the first 3D morphable face reflectance model with spatially varying BRDF using only low-cost publicly-available data. We apply linear shiness weighting into parametric modeling to represent spatially varying specular intensity and shiness. Then an inverse rendering algorithm is developed to reconstruct the reflectance parameters from non-Light Stage data, which are used to train an initial morphable reflectance model. To enhance the model's generalization capability and expressive power, we further propose an update-by-reconstruction strategy to finetune it on an in-the-wild dataset. Experimental results show that our method obtains decent rendering results with plausible facial specularities. Our code is released here.

        ----

        ## [821] StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer

        **Authors**: *Sasikarn Khwanmuang, Pakkapon Phongthawee, Patsorn Sangkloy, Supasorn Suwajanakorn*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00832](https://doi.org/10.1109/CVPR52729.2023.00832)

        **Abstract**:

        Our paper seeks to transfer the hairstyle of a reference image to an input photo for virtual hair tryon. We target a variety of challenges scenarios, such as transforming a long hairstyle with bangs to a pixie cut, which requires removing the existing hair and inferring how the forehead would look, or transferring partially visible hair from a hat-wearing person in a different pose. Past solutions leverage StyleGAN for hallucinating any missing parts and producing a seamless face-hair composite through so-called GAN inversion or projection. However, there remains a challenge in controlling the hallucinations to accurately transfer hairstyle and preserve the face shape and identity of the input. To overcome this, we propose a multi-view optimization framework that uses two different views of reference composites to semantically guide occluded or ambiguous regions. Our optimization shares information between two poses, which allows us to produce high fidelity and realistic results from incomplete references. Our framework produces high-quality results and outperforms prior work in a user study that consists of significantly more challenging hair transfer scenarios than previously studied. Project page: https://stylegan-salon.github.io/.

        ----

        ## [822] FaceLit: Neural 3D Relightable Faces

        **Authors**: *Anurag Ranjan, Kwang Moo Yi, Jen-Hao Rick Chang, Oncel Tuzel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00833](https://doi.org/10.1109/CVPR52729.2023.00833)

        **Abstract**:

        We propose a generative framework, FaceLit, capable of generating a 3D face that can be rendered at various user-defined lighting conditions and views, learned purely from 2D images in-the-wild without any manual annotation. Unlike existing works that require careful capture setup or human labor, we rely on off-the-shelf pose and illumination estimators. With these estimates, we incorporate the Phong reflectance model in the neural volume rendering framework. Our model learns to generate shape and material properties of a face such that, when rendered according to the natural statistics of pose and illumination, produces photorealistic face images with multiview 3D and illumination consistency. Our method enables photorealistic generation of faces with explicit illumination and view controls on multiple datasets - FFHQ, MetFaces and CelebA-HQ. We show state-of-the-art photorealism among 3D aware GANs on FFHQ dataset achieving an FID score of 3.5.

        ----

        ## [823] FitMe: Deep Photorealistic 3D Morphable Model Avatars

        **Authors**: *Alexandros Lattas, Stylianos Moschoglou, Stylianos Ploumpis, Baris Gecer, Jiankang Deng, Stefanos Zafeiriou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00834](https://doi.org/10.1109/CVPR52729.2023.00834)

        **Abstract**:

        In this paper, we introduce FitMe, a facial reflectance model and a differentiable rendering optimization pipeline, that can be used to acquire high-fidelity renderable human avatars from single or multiple images. The model consists of a multi-modal style-based generator, that captures facial appearance in terms of diffuse and specular reflectance, and a PCA-based shape model. We employ a fast differentiable rendering process that can be used in an optimization pipeline, while also achieving photorealistic facial shading. Our optimization process accurately captures both the facial reflectance and shape in high-detail, by exploiting the expressivity of the style-based latent representation and of our shape model. FitMe achieves state-of-the-art reflectance acquisition and identity preservation on single “in-the-wild” facial images, while it produces impressive scan-like results, when given multiple unconstrained facial images pertaining to the same identity. In contrast with recent implicit avatar reconstructions, FitMe requires only one minute and produces relightable mesh and texture-based avatars, that can be used by end-user applications.

        ----

        ## [824] NeuWigs: A Neural Dynamic Model for Volumetric Hair Capture and Animation

        **Authors**: *Ziyan Wang, Giljoo Nam, Tuur Stuyck, Stephen Lombardi, Chen Cao, Jason M. Saragih, Michael Zollhöfer, Jessica K. Hodgins, Christoph Lassner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00835](https://doi.org/10.1109/CVPR52729.2023.00835)

        **Abstract**:

        The capture and animation of human hair are two of the major challenges in the creation of realistic avatars for the virtual reality. Both problems are highly challenging, because hair has complex geometry and appearance and exhibits challenging motion. In this paper, we present a two-stage approach that models hair independently of the head to address these challenges in a data-driven manner. The first stage, state compression, learns a low-dimensional latent space of 3D hair states including motion and appearance via a novel autoencoder-as-a-tracker strategy. To better disentangle the hair and head in appearance learning, we employ multi-view hair segmentation masks in combination with a differentiable volumetric renderer. The second stage optimizes a novel hair dynamics model that performs temporal hair transfer based on the discovered latent codes. To enforce higher stability while driving our dynamics model, we employ the 3D point-cloud autoencoder from the compression stage for denoising of the hair state. Our model outperforms the state of the art in novel view synthesis and is capable of creating novel hair animations without relying on hair observations as a driving signal.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
Project page at https://ziyanwl.github.io/neuwigs/.

        ----

        ## [825] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation

        **Authors**: *Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, Fei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00836](https://doi.org/10.1109/CVPR52729.2023.00836)

        **Abstract**:

        Generating talking head videos through a face image and a piece of speech audio still contains many challenges. i.e., unnatural head movement, distorted expression, and identity modification. We argue that these issues are mainly caused by learning from the coupled 2D motion fields. On the other hand, explicitly using 3D information also suffers problems of stiff expression and incoherent video. We present SadTalker, which generates 3D motion coefficients (head pose, expression) of the 3DMM from audio and implicitly modulates a novel 3D-aware face render for talking head generation. To learn the realistic motion coefficients, we explicitly model the connections between audio and different types of motion coefficients individually. Precisely, we present ExpNet to learn the accurate facial expression from audio by distilling both coefficients and 3D-rendered faces. As for the head pose, we design PoseVAE via a conditional VAE to synthesize head motion in different styles. Finally, the generated 3D motion coefficients are mapped to the unsupervised 3D keypoints space of the proposed face render to synthesize the final video. We conducted extensive experiments to demonstrate the superiority of our method in terms of motion and video quality.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code and demo videos are available at https://sadtalker.github.io.

        ----

        ## [826] High-Fidelity Clothed Avatar Reconstruction from a Single Image

        **Authors**: *Tingting Liao, Xiaomei Zhang, Yuliang Xiu, Hongwei Yi, Xudong Liu, Guo-Jun Qi, Yong Zhang, Xuan Wang, Xiangyu Zhu, Zhen Lei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00837](https://doi.org/10.1109/CVPR52729.2023.00837)

        **Abstract**:

        This paper presents a framework for efficient 3D clothed avatar reconstruction. By combining the advantages of the high accuracy of optimization-based methods and the efficiency of learning-based methods, we propose a coarse-to-fine way to realize a high-fidelity clothed avatar reconstruction (CAR) from a single image. At the first stage, we use an implicit model to learn the general shape in the canonical space of a person in a learning-based way, and at the second stage, we refine the surface detail by estimating the non-rigid deformation in the posed space in an optimization way. A hyper-network is utilized to generate a good initialization so that the convergence of the optimization process is greatly accelerated. Extensive experiments on various datasets show that the proposed CAR successfully produces high-fidelity avatars for arbitrarily clothed humans in real scenes. The codes will be released in https://github.com/TingtingLiao/CAR.

        ----

        ## [827] Music-Driven Group Choreography

        **Authors**: *Nhat Le, Thang Pham, Tuong Do, Erman Tjiputra, Quang D. Tran, Anh Nguyen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00838](https://doi.org/10.1109/CVPR52729.2023.00838)

        **Abstract**:

        Music-driven choreography is a challenging problem with a wide variety of industrial applications. Recently, many methods have been proposed to synthesize dance motions from music for a single dancer. However, generating dance motion for a group remains an open problem. In this paper, we present AIOZ - GDANCE, a new large-scale dataset for music-driven group dance generation. Unlike existing datasets that only support single dance, our new dataset contains group dance videos, hence supporting the study of group choreography. We propose a semi-autonomous labeling method with humans in the loop to obtain the 3D ground truth for our dataset. The proposed dataset consists of 16.7 hours of paired music and 3D motion from in-the-wild videos, covering 7 dance styles and 16 music genres. We show that naively applying single dance generation technique to creating group dance motion may lead to unsatisfactory results, such as inconsistent movements and collisions between dancers. Based on our new dataset, we propose a new method that takes an input music sequence and a set of 3D positions of dancers to efficiently produce multiple group-coherent choreographies. We propose new evaluation metrics for measuring group dance quality and perform intensive experiments to demonstrate the effectiveness of our method. Our project facilitates future research on group dance generation and is available at https://aioz-ai.github.io/AIOZ-GDANCE/.

        ----

        ## [828] Hand Avatar: Free-Pose Hand Animation and Rendering from Monocular Video

        **Authors**: *Xingyu Chen, Baoyuan Wang, Heung-Yeung Shum*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00839](https://doi.org/10.1109/CVPR52729.2023.00839)

        **Abstract**:

        We present HandAvatar, a novel representation for hand animation and rendering, which can generate smoothly compositional geometry and self-occlusion-aware texture. Specifically, we first develop a MANO-HD model as a high-resolution mesh topology to fit personalized hand shapes. Sequentially, we decompose hand geometry into per-bone rigid parts, and then re-compose paired geometry encodings to derive an across-part consistent occupancy field. As for texture modeling, we propose a self-occlusion-aware shading field (SelF). In SelF, drivable anchors are paved on the MANO-HD surface to record albedo information under a wide variety of hand poses. Moreover, directed soft occupancy is designed to describe the ray-to-surface relation, which is leveraged to generate an illumination field for the disentanglement of pose-independent albedo and pose-dependent illumination. Trained from monocular video data, our HandAvatar can perform freepose hand animation and rendering while at the same time achieving superior appearance fidelity. We also demonstrate that HandAvatar provides a route for hand appearance editing. Project website: https://seanchenxy.github.iO/HandAvatarWeb.

        ----

        ## [829] Biomechanics-Guided Facial Action Unit Detection Through Force Modeling

        **Authors**: *Zijun Cui, Chenyi Kuang, Tian Gao, Kartik Talamadupula, Qiang Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00840](https://doi.org/10.1109/CVPR52729.2023.00840)

        **Abstract**:

        Existing AU detection algorithms are mainly based on appearance information extracted from 2D images, and well-established facial biomechanics that governs 3D facial skin deformation is rarely considered. In this paper, we propose a biomechanics-guided AU detection approach, where facial muscle activation forces are modelled and are employed to predict AU activation. Specifically, our model consists of two branches: 3D physics branch and 2D image branch. In 3D physics branch, we first derive the Euler-Lagrange equation governing facial deformation. The Euler-Lagrange equation represented as an ordinary differential equation (ODE) is embedded into a differentiable ODE solver. Muscle activation forces together with other physics parameters are firstly regressed, and then are utilized to simulate 3D deformation by solving the ODE. By leveraging facial biomechanics, we obtain physically plausible facial muscle activation forces. 2D image branch compensates 3D physics branch by employing additional appearance information from 2D images. Both estimated forces and appearance features are employed for AU detection. The proposed approach achieves competitive AU detection performance on two benchmark datasets. Furthermore, by leveraging biomechanics, our approach achieves outstanding performance with reduced training data.

        ----

        ## [830] Zero-shot Pose Transfer for Unrigged Stylized 3D Characters

        **Authors**: *Jiashun Wang, Xueting Li, Sifei Liu, Shalini De Mello, Orazio Gallo, Xiaolong Wang, Jan Kautz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00841](https://doi.org/10.1109/CVPR52729.2023.00841)

        **Abstract**:

        Transferring the pose of a reference avatar to stylized 3D characters of various shapes is a fundamental task in computer graphics. Existing methods either require the stylized characters to be rigged, or they use the stylized character in the desired pose as ground truth at training. We present a zero-shot approach that requires only the widely available deformed non-stylized avatars in training, and deforms stylized characters of significantly different shapes at inference. Classical methods achieve strong generalization by deforming the mesh at the triangle level, but this requires labelled correspondences. We leverage the power of local deformation, but without requiring explicit correspondence labels. We introduce a semi-supervised shape-understanding module to bypass the need for explicit correspondences at test time, and an implicit pose deformation module that deforms individual surface points to match the target pose. Further-more, to encourage realistic and accurate deformation of stylized characters, we introduce an efficient volume-based test-time training procedure. Because it does not need rigging, nor the deformed stylized character at training time, our model generalizes to categories with scarce annotation, such as stylized quadrupeds. Extensive experiments demonstrate the effectiveness of the proposed method compared to the state-of-the-art approaches trained with comparable or more supervision. Our project page is available at https://jiashunwang.github.io/ZPT/

        ----

        ## [831] Invertible Neural Skinning

        **Authors**: *Yash Kant, Aliaksandr Siarohin, Riza Alp Güler, Menglei Chai, Jian Ren, Sergey Tulyakov, Igor Gilitschenski*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00842](https://doi.org/10.1109/CVPR52729.2023.00842)

        **Abstract**:

        Building animatable and editable models of clothed humans from raw 3D scans and poses is a challenging problem. Existing reposing methods suffer from the limited expressiveness of Linear Blend Skinning (LBS), require costly mesh extraction to generate each new pose, and typically do not preserve surface correspondences across different poses. In this work, we introduce Invertible Neural Skinning (INS) to address these shortcomings. To maintain correspondences, we propose a Pose-conditioned Invertible Network (PIN) architecture, which extends the LBS process by learning additional pose-varying deformations. Next, we combine PIN with a differentiable LBS module to build an expressive and end-to-end Invertible Neural Skinning (INS) pipeline. We demonstrate the strong performance of our method by outperforming the state-of-the-art reposing techniques on clothed humans and preserving surface correspondences, while being an order of magnitude faster. We also perform an ablation study, which shows the usefulness of our pose-conditioning formulation, and our qualitative results display that INS can rectify artefacts introduced by LBS well.

        ----

        ## [832] BEDLAM: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion

        **Authors**: *Michael J. Black, Priyanka Patel, Joachim Tesch, Jinlong Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00843](https://doi.org/10.1109/CVPR52729.2023.00843)

        **Abstract**:

        We show, for the first time, that neural networks trained only on synthetic data achieve state-of-the-art accuracy on the problem of 3D human pose and shape (HPS) estimation from real images. Previous synthetic datasets have been small, unrealistic, or lacked realistic clothing. Achieving sufficient realism is non-trivial and we show how to do this for full bodies in motion. Specifically, our BEDLAM dataset contains monocular RGB videos with ground-truth 3D bodies in SMPL-X format. It includes a diversity of body shapes, motions, skin tones, hair, and clothing. The clothing is realistically simulated on the moving bodies using commercial clothing physics simulation. We render varying numbers of people in realistic scenes with varied lighting and camera motions. We then train various HPS regressors using BEDLAM and achieve state-of-the-art accuracy on real-image benchmarks despite training with synthetic data. We use BEDLAM to gain insights into what model design choices are important for accuracy. With good synthetic training data, we find that a basic method like HMR approaches the accuracy of the current SOTA method (CLIFF). BEDLAM is useful for a variety of tasks and all images, ground truth bodies, 3D clothing, support code, and more are available for research purposes. Additionally, we provide detailed information about our synthetic data generation pipeline, enabling others to generate their own datasets. See the project page: https://bedlam.is.tue.mpg.de/.

        ----

        ## [833] DIFu: Depth-Guided Implicit Function for Clothed Human Reconstruction

        **Authors**: *Dae-Young Song, HeeKyung Lee, Jeongil Seo, Donghyeon Cho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00844](https://doi.org/10.1109/CVPR52729.2023.00844)

        **Abstract**:

        Recently, implicit function (IF)-based methods for clothed human reconstruction using a single image have received a lot of attention. Most existing methods rely on a 3D embedding branch using volume such as the skinned multi-person linear (SMPL) model, to compensate for the lack of information in a single image. Beyond the SMPL, which provides skinned parametric human 3D information, in this paper, we propose a new IF-based method, DIFu, that utilizes a projected depth prior containing textured and non-parametric human 3D information. In particular, DIFu consists of a generator, an occupancy prediction network, and a texture prediction network. The generator takes an RGB image of the human front-side as input, and hallucinates the human back-side image. After that, depth maps for front/back images are estimated and projected into 3D volume space. Finally, the occupancy prediction network extracts a pixel-aligned feature and a voxel-aligned feature through a 2D encoder and a 3D encoder, respectively, and estimates occupancy using these features. Note that voxel-aligned features are obtained from the projected depth maps, thus it can contain detailed 3D information such as hair and cloths. Also, colors of each query point are also estimated with the texture inference branch. The effectiveness of DIFu is demonstrated by comparing to recent IF-based models quantitatively and qualitatively.

        ----

        ## [834] Complete 3D Human Reconstruction from a Single Incomplete Image

        **Authors**: *Junying Wang, Jae Shin Yoon, Tuanfeng Y. Wang, Krishna Kumar Singh, Ulrich Neumann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00845](https://doi.org/10.1109/CVPR52729.2023.00845)

        **Abstract**:

        This paper presents a method to reconstruct a complete human geometry and texture from an image of a person with only partial body observed, e.g., a torso. The core challenge arises from the occlusion: there exists no pixel to reconstruct where many existing single-view human reconstruction methods are not designed to handle such invisible parts, leading to missing data in 3D. To address this challenge, we introduce a novel coarse-to-fine human reconstruction framework. For coarse reconstruction, explicit volumetric features are learned to generate a complete human geometry with 3D convolutional neural networks conditioned by a 3D body model and the style features from visible parts. An implicit network combines the learned 3D features with the high-quality surface normals enhanced from multiviews to produce fine local details, e.g., high-frequency wrinkles. Finally, we perform progressive texture inpainting to reconstruct a complete appearance of the person in a view-consistent way, which is not possible without the reconstruction of a complete geometry. In experiments, we demonstrate that our method can reconstruct high-quality 3D humans, which is robust to occlusion.

        ----

        ## [835] Learning Neural Volumetric Representations of Dynamic Humans in Minutes

        **Authors**: *Chen Geng, Sida Peng, Zhen Xu, Hujun Bao, Xiaowei Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00846](https://doi.org/10.1109/CVPR52729.2023.00846)

        **Abstract**:

        This paper addresses the challenge of efficiently reconstructing volumetric videos of dynamic humans from sparse multi-view videos. Some recent works represent a dynamic human as a canonical neural radiance field (NeRF) and a motion field, which are learned from input videos through differentiable rendering. But the per-scene optimization generally requires hours. Other generalizable NeRF models leverage learned prior from datasets to reduce the optimization time by only finetuning on new scenes at the cost of visual fidelity. In this paper, we propose a novel method for learning neural volumetric representations of dynamic humans in minutes with competitive visual quality. Specifically, we define a novel part-based voxelized human representation to better distribute the representational power of the network to different human parts. Furthermore, we propose a novel 2D motion parameterization scheme to increase the convergence rate of deformation field learning. Experiments demonstrate that our model can be learned 100 times faster than previous per-scene optimization methods while being competitive in the rendering quality. Training our model on a 512 × 512 video with 100 frames typically takes about 5 minutes on a single RTX 3090 GPU. The code is available on our project page: https://zju3dv.github.io/instant_nvr.

        ----

        ## [836] Marching-Primitives: Shape Abstraction from Signed Distance Function

        **Authors**: *Weixiao Liu, Yuwei Wu, Sipu Ruan, Gregory S. Chirikjian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00847](https://doi.org/10.1109/CVPR52729.2023.00847)

        **Abstract**:

        Representing complex objects with basic geometric primitives has long been a topic in computer vision. Primitive-based representations have the merits of compactness and computational efficiency in higher-level tasks such as physics simulation, collision checking, and robotic manipulation. Unlike previous works which extract polygonal meshes from a signed distance function (SDF), in this paper, we present a novel method, named Marching-Primitives, to obtain a primitive-based abstraction directly from an SDF. Our method grows geometric primitives (such as superquadrics) iteratively by analyzing the connectivity of voxels while marching at different levels of signed distance. For each valid connected volume of interest, we march on the scope of voxels from which a primitive is able to be extracted in a probabilistic sense and simultaneously solve for the parameters of the primitive to capture the underlying local geometry. We evaluate the performance of our method on both synthetic and real-world datasets. The results show that the proposed method out-performs the state-of-the-art in terms of accuracy, and is directly generalizable among different categories and scales. The code is open-sourced at https://github.com/ChirikjianLab/Marching-Primitives.git.

        ----

        ## [837] Learning Analytical Posterior Probability for Human Mesh Recovery

        **Authors**: *Qi Fang, Kang Chen, Yinghui Fan, Qing Shuai, Jiefeng Li, Weidong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00848](https://doi.org/10.1109/CVPR52729.2023.00848)

        **Abstract**:

        Despite various probabilistic methods for modeling the uncertainty and ambiguity in human mesh recovery, their overall precision is limited because existing formulations for joint rotations are either not constrained to SO(3) or difficult to learn for neural networks. To address such an issue, we derive a novel analytical formulation for learning posterior probability distributions of human joint rotations conditioned on bone directions in a Bayesian manner, and based on this, we propose a new posterior-guided framework for human mesh recovery. We demonstrate that our framework is not only superior to existing SOTA baselines on multiple benchmarks but also flexible enough to seamlessly incorporate with additional sensors due to its Bayesian nature. The code is available at https://github.com/NetEase-GameAI/ProPose.

        ----

        ## [838] MagicPony: Learning Articulated 3D Animals in the Wild

        **Authors**: *Shangzhe Wu, Ruining Li, Tomas Jakab, Christian Rupprecht, Andrea Vedaldi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00849](https://doi.org/10.1109/CVPR52729.2023.00849)

        **Abstract**:

        We consider the problem of predicting the 3D shape, articulation, viewpoint, texture, and lighting of an articulated animal like a horse given a single test image as input. We present a new method, dubbed MagicPony, that learns this predictor purely from in-the-wild single-view images of the object category, with minimal assumptions about the topology of deformation. At its core is an implicit-explicit representation of articulated shape and appearance, combining the strengths of neural fields and meshes. In order to help the model understand an object's shape and pose, we distil the knowledge captured by an off-the-shelf self-supervised vision transformer and fuse it into the 3D model. To overcome local optima in viewpoint estimation, we further introduce a new viewpoint sampling scheme that comes at no additional training cost. MagicPony outperforms prior work on this challenging task and demonstrates excellent generalisation in reconstructing art, despite the fact that it is only trained on real images. The code can be found on the project page at https://3dmagicpony.github.io/

        ----

        ## [839] Visual-Tactile Sensing for In-Hand Object Reconstruction

        **Authors**: *Wenqiang Xu, Zhenjun Yu, Han Xue, Ruolin Ye, Siqiong Yao, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00850](https://doi.org/10.1109/CVPR52729.2023.00850)

        **Abstract**:

        Tactile sensing is one of the modalities humans rely on heavily to perceive the world. Working with vision, this modality refines local geometry structure, measures defor-mation at the contact area, and indicates the hand-object contact state. With the availability of open-source tactile sensors such as DIGIT, research on visual-tactile learning is becoming more accessible and reproducible. Leveraging this tactile sensor, we propose a novel visual-tactile in-hand object reconstruction framework VTacO, and ex-tend it to VTacOH for hand-object reconstruction. Since our method can support both rigid and deformable ob-ject reconstruction, no existing benchmarks are proper for the goal. We propose a simulation environment, VT-Sim, which supports generating hand-object interaction for both rigid and deformable objects. With VT-Sim, we gener-ate a large-scale training dataset and evaluate our method on it. Extensive experiments demonstrate that our pro-posed method can outperform the previous baseline meth-ods qualitatively and quantitatively. Finally, we directly ap-ply our model trained in simulation to various real-world test cases, which display qualitative results. Codes, mod-els, simulation environment, and datasets are available at https://sites.google.com/view/vtaco/.

        ----

        ## [840] Command-driven Articulated Object Understanding and Manipulation

        **Authors**: *Ruihang Chu, Zhengzhe Liu, Xiaoqing Ye, Xiao Tan, Xiaojuan Qi, Chi-Wing Fu, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00851](https://doi.org/10.1109/CVPR52729.2023.00851)

        **Abstract**:

        We present Cart, a new approach towards articulatedobject manipulations by human commands. Beyond the existing work that focuses on inferring articulation structures, we further support manipulating articulated shapes to align them subject to simple command templates. The key of Cart is to utilize the prediction of object structures to connect visual observations with user commands for effective manipulations. It is achieved by encoding command messages for motion prediction and a test-time adaptation to adjust the amount of movement from only command supervision. For a rich variety of object categories, Cart can accurately manipulate object shapes and outperform the state-of-the-art approaches in understanding the inherent articulation structures. Also, it can well generalize to unseen object categories and real-world objects. We hope Cart could open new directions for instructing machines to operate articulated objects. Code is available at https://github.com/dvlab-research/Cart.

        ----

        ## [841] Target-referenced Reactive Grasping for Dynamic Objects

        **Authors**: *Jirong Liu, Ruo Zhang, Haoshu Fang, Minghao Gou, Hongjie Fang, Chenxi Wang, Sheng Xu, Hengxu Yan, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00852](https://doi.org/10.1109/CVPR52729.2023.00852)

        **Abstract**:

        Reactive grasping, which enables the robot to successfully grasp dynamic moving objects, is of great interest in robotics. Current methods mainly focus on the temporal smoothness of the predicted grasp poses but few consider their semantic consistency. Consequently, the predicted grasps are not guaranteed to fall on the same part of the same object, especially in cluttered scenes. In this paper, we propose to solve reactive grasping in a target-referenced setting by tracking through generated grasp spaces. Given a targeted grasp pose on an object and detected grasp poses in a new observation, our method is composed of two stages: 1) discovering grasp pose correspondences through an attentional graph neural network and selecting the one with the highest similarity with respect to the target pose; 2) refining the selected grasp poses based on target and historical information. We evaluate our method on a large-scale benchmark GraspNet-1Billion. We also collect 30 scenes of dynamic objects for testing. The results suggest that our method outperforms other representative methods. Furthermore, our real robot experiments achieve an average success rate of over 80 percent. Code and demos are available at: https://graspnet.net/reactive.

        ----

        ## [842] NeuralDome: A Neural Modeling Pipeline on Multi-View Human-Object Interactions

        **Authors**: *Juze Zhang, Haimin Luo, Hongdi Yang, Xinru Xu, Qianyang Wu, Ye Shi, Jingyi Yu, Lan Xu, Jingya Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00853](https://doi.org/10.1109/CVPR52729.2023.00853)

        **Abstract**:

        Humans constantly interact with objects in daily life tasks. Capturing such processes and subsequently conducting visual inferences from a fixed viewpoint suffers from occlusions, shape and texture ambiguities, motions, etc. To mitigate the problem, it is essential to build a training dataset that captures free-viewpoint interactions. We construct a dense multi-view dome to acquire a complex human object interaction dataset, named HODome, that consists of ~71 M frames on 10 subjects interacting with 23 objects. To process the HODome dataset, we develop NeuralDome, a layer-wise neural processing pipeline tailored for multi-view video inputs to conduct accurate tracking, geometry reconstruction and free-view rendering, for both human subjects and objects. Extensive experiments on the HODome dataset demonstrate the effectiveness of NeuralDome on a variety of inference, modeling, and rendering tasks. Both the dataset and the NeuralDome tools will be disseminated to the community for further development, which can be found at https://juzezhang.github.io/NeuralDome

        ----

        ## [843] A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image

        **Authors**: *Changlong Jiang, Yang Xiao, Cunlin Wu, Mingyang Zhang, Jinghong Zheng, Zhiguo Cao, Joey Tianyi Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00854](https://doi.org/10.1109/CVPR52729.2023.00854)

        **Abstract**:

        3D interacting hand pose estimation from a single RGB image is a challenging task, due to serious self-occlusion and inter-occlusion towards hands, confusing similar appearance patterns between 2 hands, ill-posed joint position mapping from 2D to 3D, etc.. To address these, we propose to extend A2J-the state-of-the-art depth-based 3D single hand pose estimation method-to RGB domain under interacting hand condition. Our key idea is to equip A2J with strong local-global aware ability to well capture interacting hands' local fine details and global articulated clues among joints jointly. To this end, A2J is evolved under Transformer's non-local encoding-decoding framework to build A2J- Transformer. It holds 3 main advantages over A2J. First, self-attention across local anchor points is built to make them global spatial context aware to better capture joints' articulation clues for resisting occlusion. Secondly, each anchor point is regarded as learnable query with adaptive feature learning for facilitating pattern fitting capacity, instead of having the same local representation with the others. Last but not least, anchor point locates in 3D space instead of 2D as in A2J, to leverage 3D pose prediction. Experiments on challenging InterHand 2.6M demonstrate that, A2J-Transformer can achieve state-of-the-art model-free performance (3.38mm MPJPE advancement in 2-hand case) and can also be applied to depth domain with strong generalization. The code is avaliable at https://github.com/ChanglongJiangGit/A2J-Transformer.

        ----

        ## [844] TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments

        **Authors**: *Yu Sun, Qian Bao, Wu Liu, Tao Mei, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00855](https://doi.org/10.1109/CVPR52729.2023.00855)

        **Abstract**:

        Although the estimation of 3D human pose and shape (HPS) is rapidly progressing, current methods still cannot reliably estimate moving humans in global coordinates, which is critical for many applications. This is particularly challenging when the camera is also moving, entangling human and camera motion. To address these issues, we adopt a novel 5D representation (space, time, and identity) that enables end-to-end reasoning about people in scenes. Our method, called TRACE, introduces several novel architectural components. Most importantly, it uses two new “maps” to reason about the 3D trajectory of people over time in camera, and world, coordinates. An additional memory unit enables persistent tracking of people even during long occlusions. TRACE is the first one-stage method to jointly recover and track 3D humans in global coordinates from dynamic cameras. By training it end-to-end, and using full image information, TRACE achieves state-of-the-art performance on tracking and HPS benchmarks. The code
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://www.yusun.work/TRACE/TRACE.html and dataset
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
https://github.com/Arthur151/DynaCam are released for research purposes.

        ----

        ## [845] BITE: Beyond Priors for Improved Three-D Dog Pose Estimation

        **Authors**: *Nadine Rüegg, Shashank Tripathi, Konrad Schindler, Michael J. Black, Silvia Zuffi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00856](https://doi.org/10.1109/CVPR52729.2023.00856)

        **Abstract**:

        We address the problem of inferring the 3D shape and pose of dogs from images. Given the lack of 3D training data, this problem is challenging, and the best methods lag behind those designed to estimate human shape and pose. To make progress, we attack the problem from multiple sides at once. First, we need a good 3D shape prior, like those available for humans. To that end, we learn a dog-specific 3D parametric model, called D-SMAL. Second, existing methods focus on dogs in standing poses because when they sit or lie down, their legs are self occluded and their bodies deform. Without access to a good pose prior or 3D data, we need an alternative approach. To that end, we exploit contact with the ground as a form of side information. We consider an existing large dataset of dog images and label any 3D contact of the dog with the ground. We exploit body-ground contact in estimating dog pose and find that it significantly improves results. Third, we develop a novel neural network architecture to infer and exploit this contact information. Fourth, to make progress, we have to be able to measure it. Current evaluation metrics are based on 2D features like keypoints and silhouettes, which do not directly correlate with 3D errors. To address this, we create a synthetic dataset containing rendered images of scanned 3D dogs. With these advances, our method recovers significantly better dog shape and pose than the state of the art, and we evaluate this improvement in 3D. Our code, model and test dataset are publicly available for research purposes at https://bite.is.tue.mpg.de/.

        ----

        ## [846] PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

        **Authors**: *Qitao Zhao, Ce Zheng, Mengyuan Liu, Pichao Wang, Chen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00857](https://doi.org/10.1109/CVPR52729.2023.00857)

        **Abstract**:

        Recently, transformer-based methods have gained significant success in sequential 2D-to-3D lifting human pose estimation. As a pioneering work, PoseFormer captures spatial relations of human joints in each video frame and human dynamics across frames with cascaded transformer layers and has achieved impressive performance. However, in real scenarios, the performance of PoseFormer and its follow-ups is limited by two factors: (a) The length of the input joint sequence; (b) The quality of 2D joint detection. Existing methods typically apply self-attention to all frames of the input sequence, causing a huge computational burden when the frame number is increased to obtain advanced estimation accuracy, and they are not robust to noise naturally brought by the limited capability of 2D joint detectors. In this paper, we propose PoseFormerV2, which exploits a compact representation of lengthy skeleton sequences in the frequency domain to efficiently scale up the receptive field and boost robustness to noisy 2D joint detection. With minimum modifications to PoseFormer, the proposed method effectively fuses features both in the time domain and frequency domain, enjoying a better speed-accuracy trade-off than its precursor. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that the proposed approach significantly outperforms the original PoseFormer and other transformer-based variants. Code is released at https://github.com/ QitaoZhao/PoseFormerV2.

        ----

        ## [847] Global-to-Local Modeling for Video-Based 3D Human Pose and Shape Estimation

        **Authors**: *Xiaolong Shen, Zongxin Yang, Xiaohan Wang, Jianxin Ma, Chang Zhou, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00858](https://doi.org/10.1109/CVPR52729.2023.00858)

        **Abstract**:

        Video-based 3D human pose and shape estimations are evaluated by intra-frame accuracy and inter-frame smoothness. Although these two metrics are responsible for different ranges of temporal consistency, existing state-of-the-art methods treat them as a unified problem and use monotonous modeling structures (e.g., RNN or attention-based block) to design their networks. However, using a single kind of modeling structure is difficult to balance the learning of short-term and long-term temporal correlations, and may bias the network to one of them, leading to undesirable predictions like global location shift, temporal inconsistency, and insufficient local details. To solve these problems, we propose to structurally decouple the modeling of long-term and short-term correlations in an end-to-end framework, Global-to-Local Transformer (GLoT). First, a global transformer is introduced with a Masked Pose and Shape Estimation strategy for long-term modeling. The strategy stimulates the global transformer to learn more inter-frame correlations by randomly masking the features of several frames. Second, a local transformer is responsible for exploiting local details on the human mesh and interacting with the global transformer by leveraging cross-attention. Moreover, a Hierarchical Spatial Correlation Regressor is further introduced to refine intra-frame estimations by decoupled global-local representation and implicit kinematic constraints. Our GLoT surpasses previous state-of-the-art methods with the lowest model parameters on popular benchmarks, i.e., 3DPW, MPI-INF-3DHP, and Human3.6M. Codes are available at https://github.com/sxl142/GLoT.

        ----

        ## [848] TokenHPE: Learning Orientation Tokens for Efficient Head Pose Estimation via Transformers

        **Authors**: *Cheng Zhang, Hai Liu, Yongjian Deng, Bochen Xie, Youfu Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00859](https://doi.org/10.1109/CVPR52729.2023.00859)

        **Abstract**:

        Head pose estimation (HPE) has been widely used in the fields of human machine interaction, self-driving, and attention estimation. However, existing methods cannot deal with extreme head pose randomness and serious occlusions. To address these challenges, we identify three cues from head images, namely, neighborhood similarities, significant facial changes, and critical minority relationships. To leverage the observed findings, we propose a novel critical minority relationship-aware method based on the Transformer architecture in which the facial part relationships can be learned. Specifically, we design several orientation tokens to explicitly encode the basic orientation regions. Meanwhile, a novel token guide multiloss function is designed to guide the orientation tokens as they learn the desired regional similarities and relationships. We evaluate the proposed method on three challenging benchmark HPE datasets. Experiments show that our method achieves better performance compared with state-of-the-art methods. Our code is publicly available at https://github.com/zc2023/TokenHPE.

        ----

        ## [849] GFIE: A Dataset and Baseline for Gaze-Following from 2D to 3D in Indoor Environments

        **Authors**: *Zhengxi Hu, Yuxue Yang, Xiaolin Zhai, Dingye Yang, Bohan Zhou, Jingtai Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00860](https://doi.org/10.1109/CVPR52729.2023.00860)

        **Abstract**:

        Gaze-following is a kind of research that requires locating where the person in the scene is looking automatically under the topic of gaze estimation. It is an important clue for understanding human intention, such as identifying objects or regions of interest to humans. However, a survey of datasets used for gaze-following tasks reveals defects in the way they collect gaze point labels. Manual labeling may introduce subjective bias and is labor-intensive, while automatic labeling with an eye-tracking device would alter the person's appearance. In this work, we introduce GFIE, a novel dataset recorded by a gaze data collection system we developed. The system is constructed with two devices, an Azure Kinect and a laser rangefinder, which generate the laser spot to steer the subject's attention as they perform in front of the camera. And an algorithm is developed to locate laser spots in images for annotating 2D/3D gaze targets and removing ground truth introduced by the spots. The whole procedure of collecting gaze behavior allows us to obtain unbiased labels in unconstrained environments semi-automatically. We also propose a baseline method with stereo field-of-view (FoV) perception for establishing a 2D/3D gaze-following benchmark on the GFIE dataset. Project page: https://sites.google.com/view/gfie.

        ----

        ## [850] Robot Structure Prior Guided Temporal Attention for Camera-to-Robot Pose Estimation from Image Sequence

        **Authors**: *Yang Tian, Jiyao Zhang, Zekai Yin, Hao Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00861](https://doi.org/10.1109/CVPR52729.2023.00861)

        **Abstract**:

        In this work, we tackle the problem of online camera-to-robot pose estimation from single-view successive frames of an image sequence, a crucial task for robots to interact with the world. The primary obstacles of this task are the robot's self-occlusions and the ambiguity of single-view images. This work demonstrates, for the first time, the effectiveness of temporal information and the robot structure prior in addressing these challenges. Given the successive frames and the robot joint configuration, our method learns to accurately regress the 2D coordinates of the predefined robot's keypoints (e.g. joints). With the camera intrinsic and robotic joints status known, we get the camera-to-robot pose using a Perspective-n-point (PnP) solver. We further improve the camera-to-robot pose iteratively using the robot structure prior. To train the whole pipeline, we build a large-scale synthetic dataset generated with domain randomisation to bridge the sim-to-real gap. The extensive experiments on synthetic and real-world datasets and the downstream robotic grasping task demonstrate that our method achieves new state-of-the-art performances and outperforms traditional hand-eye calibration algorithms in real-time (36 FPS). Code and data are available at the project page: https://sites.google.com/view/sgtapose.

        ----

        ## [851] Rigidity-Aware Detection for 6D Object Pose Estimation

        **Authors**: *Yang Hai, Rui Song, Jiaojiao Li, Mathieu Salzmann, Yinlin Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00862](https://doi.org/10.1109/CVPR52729.2023.00862)

        **Abstract**:

        Most recent 6D object pose estimation methods first use object detection to obtain 2D bounding boxes before actually regressing the pose. However, the general object detection methods they use are ill-suited to handle cluttered scenes, thus producing poor initialization to the subsequent pose network. To address this, we propose a rigidity-aware detection method exploiting the fact that, in 6D pose estimation, the target objects are rigid. This lets us introduce an approach to sampling positive object regions from the entire visible object area during training, instead of naively drawing samples from the bounding box center where the object might be occluded. As such, every visible object part can contribute to the final bounding box prediction, yielding better detection robustness. Key to the success of our approach is a visibility map, which we propose to build using a minimum barrier distance between every pixel in the bounding box and the box boundary. Our results on seven challenging 6D pose estimation datasets evidence that our method outperforms general detection frameworks by a large margin. Furthermore, combined with a pose regression network, we obtain state-of-the-art pose estimation results on the challenging BOP benchmark.

        ----

        ## [852] Crowd3D: Towards Hundreds of People Reconstruction from a Single Image

        **Authors**: *Hao Wen, Jing Huang, Huili Cui, Haozhe Lin, Yu-Kun Lai, Lu Fang, Kun Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00863](https://doi.org/10.1109/CVPR52729.2023.00863)

        **Abstract**:

        Image-based multi-person reconstruction in wide-field large scenes is critical for crowd analysis and security alert. However, existing methods cannot deal with large scenes containing hundreds of people, which encounter the challenges of large number of people, large variations in human scale, and complex spatial distribution. In this paper, we propose Crowd3D, the first framework to reconstruct the 3D poses, shapes and locations of hundreds of people with global consistency from a single large-scene image. The core of our approach is to convert the problem of complex crowd localization into pixel localization with the help of our newly defined concept, Human-scene Virtual Interaction Point (HVIP). To reconstruct the crowd with global consistency, we propose a progressive reconstruction network based on HVIP by pre-estimating a scene-level camera and a ground plane. To deal with a large number of persons and various human sizes, we also design an adaptive human-centric cropping scheme. Besides, we contribute a benchmark dataset, LargeCrowd, for crowd reconstruction in a large scene. Experimental results demonstrate the effectiveness of the proposed method. The code and the dataset are available at http://cic.tju.edu.cn/faculty/likun/projects/Crowd3D.

        ----

        ## [853] Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation

        **Authors**: *Heng Yang, Marco Pavone*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00864](https://doi.org/10.1109/CVPR52729.2023.00864)

        **Abstract**:

        The two-stage object pose estimation paradigm first detects semantic keypoints on the image and then estimates the 6D pose by minimizing reprojection errors. Despite performing well on standard benchmarks, existing techniques offer no provable guarantees on the quality and uncertainty of the estimation. In this paper, we inject two fundamental changes, namely conformal keypoint detection and geometric uncertainty propagation, into the two-stage paradigm and propose the first pose estimator that endows an estimation with provable and computable worst-case error bounds. On one hand, conformal keypoint detection applies the statistical machinery of inductive conformal prediction to convert heuristic keypoint detections into circular or elliptical prediction sets that cover the groundtruth keypoints with a user-specified marginal probability (e.g., 90%). Geometric uncertainty propagation, on the other, propagates the geometric constraints on the keypoints to the 6D object pose, leading to a Pose UnceRtainty SEt (PURSE) that guarantees coverage of the groundtruth pose with the same probability. The PURSE, however, is a nonconvex set that does not directly lead to estimated poses and uncertainties. Therefore, we develop RANdom SAmple averaGing (RANSAG) to compute an average pose and apply semidefinite relaxation to upper bound the worst-case errors between the average pose and the groundtruth. On the LineMOD Occlusion dataset we demonstrate: (i) the PURSE covers the groundtruth with valid probabilities; (ii) the worst-case error bounds provide correct uncertainty quantification; and (iii) the average pose achieves better or similar accuracy as representative methods based on sparse keypoints.

        ----

        ## [854] expOSE: Accurate Initialization-Free Projective Factorization using Exponential Regularization

        **Authors**: *José Pedro Iglesias, Amanda Nilsson, Carl Olsson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00865](https://doi.org/10.1109/CVPR52729.2023.00865)

        **Abstract**:

        Bundle adjustment is a key component in practically all available Structure from Motion systems. While it is crucial for achieving accurate reconstruction, convergence to the right solution hinges on good initialization. The recently introduced factorization-based pOSE methods formulate a surrogate for the bundle adjustment error without reliance on good initialization. In this paper, we show that pOSE has an undesirable penalization of large depths. To address this we propose expOSE which has an exponential regularization that is negligible for positive depths. To achieve efficient inference we use a quadratic approximation that allows an iterative solution with VarPro. Furthermore, we extend the method with radial distortion robustness by decomposing the Object Space Error into radial and tangential components. Experimental results confirm that the proposed method is robust to initialization and improves reconstruction quality compared to state-of-the-art methods even without bundle adjustment refinement.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

        ----

        ## [855] Neural Voting Field for Camera-Space 3D Hand Pose Estimation

        **Authors**: *Lin Huang, Chung-Ching Lin, Kevin Lin, Lin Liang, Lijuan Wang, Junsong Yuan, Zicheng Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00866](https://doi.org/10.1109/CVPR52729.2023.00866)

        **Abstract**:

        We present a unified framework for camera-space 3D hand pose estimation from a single RGB image based on 3D implicit representation. As opposed to recent works, most of which first adopt holistic or pixel-level dense regression to obtain relative 3D hand pose and then follow with complex second-stage operations for 3D global root or scale recovery, we propose a novel unified 3D dense regression scheme to estimate camera-space 3D hand pose via dense 3D point-wise voting in camera frustum. Through direct dense modeling in 3D domain inspired by Pixel-aligned Implicit Functions for 3D detailed reconstruction, our proposed Neural Voting Field (NVF) fully models 3D dense local evidence and hand global geometry, helping to alleviate common 2D-to-3D ambiguities. Specifically, for a 3D query point in camera frustum and its pixel-aligned image feature, NVF, represented by a Multi-Layer Perceptron, regresses: (i) its signed distance to the hand surface; (ii) a set of 4D offset vectors (1D voting weight and 3D directional vector to each hand joint). Following a vote-casting scheme, 4D offset vectors from near-surface points are selected to calculate the 3D hand joint coordinates by a weighted average. Experiments demonstrate that NVF outperforms existing state-of-the-art algorithms on FreiHAND dataset for camera-space 3D hand pose estimation. We also adapt NVF to the classic task of root-relative 3D hand pose estimation, for which NVF also obtains state-of-the-art results on HO3D dataset.

        ----

        ## [856] Two-View Geometry Scoring Without Correspondences

        **Authors**: *Axel Barroso-Laguna, Eric Brachmann, Victor Adrian Prisacariu, Gabriel J. Brostow, Daniyar Turmukhambetov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00867](https://doi.org/10.1109/CVPR52729.2023.00867)

        **Abstract**:

        Camera pose estimation for two-view geometry traditionally relies on RANSAC. Normally, a multitude of image correspondences leads to a pool of proposed hypotheses, which are then scored to find a winning model. The inlier count is generally regarded as a reliable indicator of “consensus”. We examine this scoring heuristic, and find that it favors disappointing models under certain circumstances. As a remedy, we propose the Fundamental Scoring Network (FSNet), which infers a score for a pair of overlap-ping images and any proposed fundamental matrix. It does not rely on sparse correspondences, but rather embodies a two-view geometry model through an epipolar attention mechanism that predicts the pose error of the two images. FSNet can be incorporated into traditional RANSAC loops. We evaluate FSNet onfundamental and essential matrix estimation on indoor and outdoor datasets, and establish that FSNet can successfully identify good poses for pairs of images with few or unreliable correspondences. Besides, we show that naively combining FSNet with MAGSAC++ scoring approach achieves state of the art results.

        ----

        ## [857] Four-view Geometry with Unknown Radial Distortion

        **Authors**: *Petr Hruby, Viktor Korotynskiy, Timothy Duff, Luke Oeding, Marc Pollefeys, Tomás Pajdla, Viktor Larsson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00868](https://doi.org/10.1109/CVPR52729.2023.00868)

        **Abstract**:

        We present novel solutions to previously unsolved prob-lems of relative pose estimation from images whose calibration parameters, namely focal lengths and radial distortion, are unknown. Our approach enables metric reconstruction without modeling these parameters. The minimal case for reconstruction requires 13 points in 4 views for both the calibrated and uncalibrated cameras. We describe and implement the first solution to these minimal problems. In the calibrated case, this may be modeled as a polynomial sys-tem of equations with 3584 solutions. Despite the apparent intractability, the problem decomposes spectacularly. Each solution falls into a Euclidean symmetry class of size 16, and we can estimate 224 class representatives by solving a sequence of three subproblems with 28, 2, and 4 solutions. We highlight the relationship between internal constraints on the radial quadrifocal tensor and the relations among the principal minors of a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$4\times 4$</tex>
 matrix. We also address the case of 4 upright cameras, where 7 points are minimal. Finally, we evaluate our approach on simulated and real data and benchmark against previous calibration-free solutions, and show that our method provides an efficient startup for an SfM pipeline with radial cameras.

        ----

        ## [858] BKinD-3D: Self-Supervised 3D Keypoint Discovery from Multi-View Videos

        **Authors**: *Jennifer J. Sun, Lili Karashchuk, Amil Dravid, Serim Ryou, Sonia Fereidooni, John C. Tuthill, Aggelos K. Katsaggelos, Bingni W. Brunton, Georgia Gkioxari, Ann Kennedy, Yisong Yue, Pietro Perona*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00869](https://doi.org/10.1109/CVPR52729.2023.00869)

        **Abstract**:

        Quantifying motion in 3D is important for studying the behavior of humans and other animals, but manual pose annotations are expensive and time-consuming to obtain. Self-supervised keypoint discovery is a promising strategy for estimating 3D poses without annotations. However, current keypoint discovery approaches commonly process single 2D views and do not operate in the 3D space. We propose a new method to perform self-supervised keypoint discovery in 3D from multi-view videos of behaving agents, without any keypoint or bounding box supervision in 2D or 3D. Our method, BKinD-3D, uses an encoder-decoder architecture with a 3D volumetric heatmap, trained to reconstruct spatiotemporal differences across multiple views, in addition to joint length constraints on a learned 3D skeleton of the subject. In this way, we discover keypoints without requiring manual supervision in videos of humans and rats, demonstrating the potential of 3D keypoint discovery for studying behavior.

        ----

        ## [859] BAAM: Monocular 3D pose and shape reconstruction with bi-contextual attention module and attention-guided modeling

        **Authors**: *Hyo-Jun Lee, Hanul Kim, Su-Min Choi, Seong-Gyun Jeong, Yeong Jun Koh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00870](https://doi.org/10.1109/CVPR52729.2023.00870)

        **Abstract**:

        3D traffic scene comprises various 3D information about car objects, including their pose and shape. However, most recent studies pay relatively less attention to reconstructing detailed shapes. Furthermore, most of them treat each 3D object as an independent one, resulting in losses of relative context inter-objects and scene context reflecting road circumstances. A novel monocular 3D pose and shape reconstruction algorithm, based on bi-contextual attention and attention-guided modeling (BAAM), is proposed in this work. First, given 2D primitives, we reconstruct 3D object shape based on attention-guided modeling that considers the relevance between detected objects and vehicle shape priors. Next, we estimate 3D object pose through bi-contextual attention, which leverages relation-context inter objects and scene-context between an object and road environment. Finally, we propose a 3D nonmaximum suppression algorithm to eliminate spurious objects based on their Bird-Eye-View distance. Extensive experiments demonstrate that the proposed BAAM yields state-of-the-art performance on ApolloCar3D. Also, they show that the proposed BAAM can be plugged into any mature monocular 3D object detector on KITTI and significantly boost their performance. Code is available at https://github.com/gywns6287/BAAM.

        ----

        ## [860] Multi-Object Manipulation via Object-Centric Neural Scattering Functions

        **Authors**: *Stephen Tian, Yancheng Cai, Hong-Xing Yu, Sergey Zakharov, Katherine Liu, Adrien Gaidon, Yunzhu Li, Jiajun Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00871](https://doi.org/10.1109/CVPR52729.2023.00871)

        **Abstract**:

        Learned visual dynamics models have proven effective for robotic manipulation tasks. Yet, it remains unclear how best to represent scenes involving multi-object interactions. Current methods decompose a scene into discrete objects, but they struggle with precise modeling and manipulation amid challenging lighting conditions as they only encode appearance tied with specific illuminations. In this work, we propose using object-centric neural scattering functions (OSFs) as object representations in a model-predictive control framework. OSFs model per-object light transport, enabling compositional scene re-rendering under object rearrangement and varying lighting conditions. By combining this approach with inverse parameter estimation and graph-based neural dynamics models, we demonstrate improved model-predictive control performance and generalization in compositional multi-object environments, even in previously unseen scenarios and harsh lighting conditions.

        ----

        ## [861] Neural Part Priors: Learning to Optimize Part-Based Object Completion in RGB-D Scans

        **Authors**: *Aleksei Bokhovkin, Angela Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00872](https://doi.org/10.1109/CVPR52729.2023.00872)

        **Abstract**:

        3D scene understanding has seen significant advances in recent years, but has largely focused on object understanding in 3D scenes with independent per-object predictions. We thus propose to learn Neural Part Priors (NPPs), parametric spaces of objects and their parts, that enable optimizing to fit to a new input 3D scan with global scene consistency constraints. The rich structure of our NPPs enables accurate, holistic scene reconstruction across similar objects in the scene. Both objects and their part geometries are characterized by coordinate field MLPs, facilitating optimization at test time to fit to input geometric observations as well as similar objects in the input scan. This enables more accurate reconstructions than independent per-object predictions as a single forward pass, while establishing global consistency within a scene. Experiments on the ScanNet dataset demonstrate that NP Ps significantly outperforms the state-of-the-art in part decomposition and object completion in real-world scenes.

        ----

        ## [862] Panoptic Lifting for 3D Scene Understanding with Neural Fields

        **Authors**: *Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Norman Müller, Matthias Nießner, Angela Dai, Peter Kontschieder*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00873](https://doi.org/10.1109/CVPR52729.2023.00873)

        **Abstract**:

        We propose Panoptic Lifting, a novel approach for learning panoptic 3D volumetric representations from images of in-the-wild scenes. Once trained, our model can render color images together with 3D-consistent panoptic segmentation from novel viewpoints. Unlike existing approaches which use 3D input directly or indirectly, our method requires only machine-generated 2D panoptic segmentation masks inferred from a pre-trained network. Our core contribution is a panoptic lifting scheme based on a neural field representation that generates a unified and multi-view consistent, 3D panoptic representation of the scene. To account for inconsistencies of 2D instance identifiers across views, we solve a linear assignment with a cost based on the model's current predictions and the machine-generated segmentation masks, thus enabling us to lift 2D instances to 3D in a consistent way. We further propose and ablate contributions that make our method more robust to noisy, machine-generated labels, including test-time augmentations for confidence estimates, segment consistency loss, bounded segmentation fields, and gradient stopping. Experimental results validate our approach on the challenging Hypersim, Replica, and ScanNet datasets, improving by 8.4, 13.8, and 10.6% in scene-level PQ over state of the art.

        ----

        ## [863] Virtual Occlusions Through Implicit Depth

        **Authors**: *Jamie Watson, Mohamed Sayed, Zawar Qureshi, Gabriel J. Brostow, Sara Vicente, Oisin Mac Aodha, Michael Firman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00874](https://doi.org/10.1109/CVPR52729.2023.00874)

        **Abstract**:

        For augmented reality (AR), it is important that virtual assets appear to ‘sit among’ real world objects. The virtual element should variously occlude and be occluded by real matter, based on a plausible depth ordering. This occlusion should be consistent over time as the viewer's camera moves. Unfortunately, small mistakes in the estimated scene depth can ruin the downstream occlusion mask, and thereby the AR illusion. Especially in real-time settings, depths inferred near boundaries or across time can be inconsistent. In this paper, we challenge the need for depth-regression as an intermediate step. We instead propose an implicit model for depth and use that to predict the occlusion mask directly. The inputs to our network are one or more color images, plus the known depths of any virtual geometry. We show how our occlusion predictions are more accurate and more temporally stable than predictions derived from traditional depth-estimation models. We obtain state-of-the-art occlusion results on the challenging ScanNetv2 dataset and superior qualitative results on real scenes.

        ----

        ## [864] Multiview Compressive Coding for 3D Reconstruction

        **Authors**: *Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, Georgia Gkioxari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00875](https://doi.org/10.1109/CVPR52729.2023.00875)

        **Abstract**:

        A central goal of visual recognition is to understand objects and scenes from a single image. 2D recognition has witnessed tremendous progress thanks to large-scale learning and general-purpose representations. Comparatively, 3D poses new challenges stemming from occlusions not depicted in the image. Prior works try to overcome these by inferring from multiple views or rely on scarce CAD models and category-specific priors which hinder scaling to novel settings. In this work, we explore single-view 3D reconstruction by learning generalizable representations inspired by advances in self-supervised learning. We introduce a simple framework that operates on 3D points of single objects or whole scenes coupled with category-agnostic large-scale training from diverse RGB-D videos. Our model, Multiview Compressive Coding (MCC), learns to compress the input appearance and geometry to predict the 3D structure by querying a 3D-aware decoder. MCC's generality and efficiency allow it to learn from large-scale and diverse data sources with strong generalization to novel objects imagined by DALLE2 or captured in-the-wild with an iPhone.

        ----

        ## [865] Behind the Scenes: Density Fields for Single View Reconstruction

        **Authors**: *Felix Wimbauer, Nan Yang, Christian Rupprecht, Daniel Cremers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00876](https://doi.org/10.1109/CVPR52729.2023.00876)

        **Abstract**:

        Inferring a meaningful geometric scene representation from a single image is a fundamental problem in computer vision. Approaches based on traditional depth map prediction can only reason about areas that are visible in the image. Currently, neural radiance fields (NeRFs) can capture true 3D including color, but are too complex to be generated from a single image. As an alternative, we propose to predict an implicit density field from a single image. It maps every location in the frustum of the image to volumetric density. By directly sampling color from the available views instead of storing color in the density field, our scene representation becomes significantly less complex compared to NeRFs, and a neural network can predict it in a single forward pass. The network is trained through self-supervision from only video data. Our formulation allows volume rendering to perform both depth prediction and novel view synthesis. Through experiments, we show that our method is able to predict meaningful geometry for regions that are occluded in the input image. Additionally, we demonstrate the potential of our approach on three datasets for depth prediction and novel-view synthesis.

        ----

        ## [866] VoxFormer: Sparse Voxel Transformer for Camera-Based 3D Semantic Scene Completion

        **Authors**: *Yiming Li, Zhiding Yu, Christopher B. Choy, Chaowei Xiao, José M. Álvarez, Sanja Fidler, Chen Feng, Anima Anandkumar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00877](https://doi.org/10.1109/CVPR52729.2023.00877)

        **Abstract**:

        Humans can easily imagine the complete 3D geometry of occluded objects and scenes. This appealing ability is vital for recognition and understanding. To enable such capability in AI systems, we propose VoxFormer, a Transformer-based semantic scene completion framework that can output complete 3D volumetric semantics from only 2D images. Our framework adopts a two-stage design where we start from a sparse set of visible and occupied voxel queries from depth estimation, followed by a densification stage that generates dense 3D voxels from the sparse ones. A key idea of this design is that the visual features on 2D images correspond only to the visible scene structures rather than the occluded or empty spaces. Therefore, starting with the fea-turization and prediction of the visible structures is more reliable. Once we obtain the set of sparse queries, we apply a masked autoencoder design to propagate the information to all the voxels by self-attention. Experiments on SemanticKITTI show that VoxFormer outperforms the state of the art with a relative improvement of 20.0% in geometry and 18.1% in semantics and reduces GPU memory during training to less than 16GB. Our code is available on https://github.com/NV1abs/VoxFormer.

        ----

        ## [867] Renderable Neural Radiance Map for Visual Navigation

        **Authors**: *Obin Kwon, Jeongho Park, Songhwai Oh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00878](https://doi.org/10.1109/CVPR52729.2023.00878)

        **Abstract**:

        We propose a novel type of map for visual navigation, a renderable neural radiance map (RNR-Map), which is designed to contain the overall visual information of a 3D environment. The RNR-Map has a grid form and consists of latent codes at each pixel. These latent codes are embedded from image observations, and can be converted to the neural radiance field which enables image rendering given a camera pose. The recorded latent codes implicitly contain visual information about the environment, which makes the RNR-Map visually descriptive. This visual information in RNR-Map can be a useful guideline for visual localization and navigation. We develop localization and navigation frameworks that can effectively utilize the RNR-Map. We evaluate the proposed frameworks on camera tracking, visual localization, and image-goal navigation. Experimental results show that the RNR-Map-based localization framework can find the target location based on a single query image with fast speed and competitive accuracy compared to other baselines. Also, this localization framework is robust to environmental changes, and even finds the most visually similar places when a query image from a different environment is given. The proposed navigation framework outperforms the existing image-goal navigation methods in difficult scenarios, under odometry and actuation noises. The navigation framework shows 65.7% success rate in curved scenarios of the NRNS [21] dataset, which is an improvement of 18.6% over the current state-of-the-art. Project page: https://rllab-snu.github.io/projects/RNR-Map/

        ----

        ## [868] Learning to Detect Mirrors from Videos via Dual Correspondences

        **Authors**: *Jiaying Lin, Xin Tan, Rynson W. H. Lau*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00879](https://doi.org/10.1109/CVPR52729.2023.00879)

        **Abstract**:

        Detecting mirrors from static images has received significant research interest recently. However, detecting mirrors over dynamic scenes is still under-explored due to the lack of a high-quality dataset and an effective method for video mirror detection (VMD). To the best of our knowledge, this is the first work to address the VMD problem from a deep-learning-based perspective. Our observation is that there are often correspondences between the contents inside (reflected) and outside (real) of a mirror, but such correspondences may not always appear in every frame, e.g., due to the change of camera pose. This inspires us to propose a video mirror detection method, named VMD-Net, that can tolerate spatially missing correspondences by considering the mirror correspondences at both the intra-frame level as well as inter-frame level via a dual correspondence module that looks over multiple frames spatially and temporally for correlating correspondences. We further propose a first large-scale dataset for VMD (named VMD-D), which contains 14,987 image frames from 269 videos with corresponding manually annotated masks. Experimental results show that the proposed method outperforms SOTA methods from relevant fields. To enable real-time VMD, our method efficiently utilizes the backbone features by removing the redundant multi-level module design and gets rid of post-processing of the output maps commonly used in existing methods, making it very efficient and practical for real-time video-based applications. Code, dataset, and models are available at https://jiaying.link/cvpr2023-vmd/

        ----

        ## [869] Temporally Consistent Online Depth Estimation Using Point-Based Fusion

        **Authors**: *Numair Khan, Eric Penner, Douglas Lanman, Lei Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00880](https://doi.org/10.1109/CVPR52729.2023.00880)

        **Abstract**:

        Depth estimation is an important step in many computer vision problems such as 3D reconstruction, novel view synthesis, and computational photography. Most existing work focuses on depth estimation from single frames. When applied to videos, the result lacks temporal consistency, showing flickering and swimming artifacts. In this paper we aim to estimate temporally consistent depth maps of video streams in an online setting. This is a difficult problem as future frames are not available and the method must choose between enforcing consistency and correcting errors from previous estimations. The presence of dynamic objects further complicates the problem. We propose to address these challenges by using a global point cloud that is dynamically updated each frame, along with a learned fusion approach in image space. Our approach encourages consistency while simultaneously allowing updates to handle errors and dynamic objects. Qualitative and quantitative results show that our method achieves state-of-the-art quality for consistent video depth estimation.

        ----

        ## [870] Zero-Shot Dual-Lens Super-Resolution

        **Authors**: *Ruikang Xu, Mingde Yao, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00881](https://doi.org/10.1109/CVPR52729.2023.00881)

        **Abstract**:

        The asymmetric dual-lens configuration is commonly available on mobile devices nowadays, which naturally stores a pair of wide-angle and telephoto images of the same scene to support realistic super-resolution (SR). Even on the same device, however, the degradation for modeling realistic SR is image-specific due to the unknown acquisition process (e.g., tiny camera motion). In this paper, we propose a zero-shot solution for dual-lens SR (ZeDuSR), where only the dual-lens pair at test time is used to learn an image-specific SR model. As such, ZeDuSR adapts itself to the current scene without using external training data, and thus gets rid of generalization difficulty. However, there are two major challenges to achieving this goal: 1) dual-lens alignment while keeping the realistic degradation, and 2) effective usage of highly limited training data. To overcome these two challenges, we propose a degradation-invariant alignment method and a degradation-aware training strategy to fully exploit the information within a single dual-lens pair. Extensive experiments validate the superiority of Ze-DuSR over existing solutions on both synthesized and real-world dual-lens datasets. The implementation code is available at https://github.com/XrKang/ZeDuSR.

        ----

        ## [871] Fully Self-Supervised Depth Estimation from Defocus Clue

        **Authors**: *Haozhe Si, Bin Zhao, Dong Wang, Yunpeng Gao, Mulin Chen, Zhigang Wang, Xuelong Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00882](https://doi.org/10.1109/CVPR52729.2023.00882)

        **Abstract**:

        Depth-from-defocus (DFD), modeling the relationship between depth and defocus pattern in images, has demonstrated promising performance in depth estimation. Recently, several self-supervised works try to overcome the difficulties in acquiring accurate depth ground-truth. However, they depend on the all-in-focus (AIF) images, which cannot be captured in real-world scenarios. Such limitation discourages the applications of DFD methods. To tackle this issue, we propose a completely self-supervised framework that estimates depth purely from a sparse focal stack. We show that our framework circumvents the needs for the depth and AIF image ground-truth, and receives superior predictions, thus closing the gap between the theoretical success of DFD works and their applications in the real world. In particular, we propose (i) a more realistic setting for DFD tasks, where no depth or AIF image ground-truth is available; (ii) a novel self- supervision framework that provides reliable predictions of depth and AIF image under the challenging setting. The proposed framework uses a neural model to predict the depth and AIF image, and utilizes an optical model to validate and refine the prediction. We verify our framework on three benchmark datasets with rendered focal stacks and real focal stacks. Qualitative and quantitative evaluations show that our method provides a strong baseline for self- supervised DFD tasks. The source code is publicly avail- able at https://github.com/Ehzoahis/DEReD.

        ----

        ## [872] MVImgNet: A Large-scale Dataset of Multi-view Images

        **Authors**: *Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, Guanying Chen, Shuguang Cui, Xiaoguang Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00883](https://doi.org/10.1109/CVPR52729.2023.00883)

        **Abstract**:

        Being data-driven is one of the most iconic properties of deep learning algorithms. The birth of ImageNet [24] drives a remarkable trend of ‘learning from large-scale data’ in computer vision. Pretraining on ImageNet to obtain rich universal representations has been manifested to benefit various 2D visual tasks, and becomes a standard in 2D vision. However, due to the laborious collection of real-world 3D data, there is yet no generic dataset serving as a counterpart of ImageNet in 3D vision, thus how such a dataset can impact the 3D community is unraveled. To remedy this defect, we introduce MVImgNet, a large-scale dataset of multi-view images, which is highly convenient to gain by shooting videos of real-world objects in human daily life. It contains 6.5 million frames from 219,188 videos crossing objects from 238 classes, with rich annotations of object masks, camera parameters, and point clouds. The multi-view attribute endows our dataset with 3D-aware signals, making it a soft bridge between 2D and 3D vision. We conduct pilot studies for probing the potential of MVImgNet on a variety of 3D and 2D visual tasks, including radiance field reconstruction, multi-view stereo, and view-consistent image understanding, where MVImgNet demonstrates promising performance, remaining lots of possibilities for future explorations. Besides, via dense reconstruction on MVImgNet, a 3D object point cloud dataset is derived, called MVPNet, covering 87,200 samples from 150 categories, with the class label on each point cloud. Experiments show that MVP-Net can benefit the real-world 3D object classification while posing new challenges to point cloud understanding. MVImgNet and MVPNet will be public, hoping to inspire the broader vision community.

        ----

        ## [873] Revisiting the Stack-Based Inverse Tone Mapping

        **Authors**: *Ning Zhang, Yuyao Ye, Yang Zhao, Ronggang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00884](https://doi.org/10.1109/CVPR52729.2023.00884)

        **Abstract**:

        Current stack-based inverse tone mapping (ITM) methods can recover high dynamic range (HDR) radiance by predicting a set of multi-exposure images from a single low dynamic range image. However, there are still some limitations. On the one hand, these methods estimate a fixed number of images (e.g., three exposure-up and three exposure-down), which may introduce unnecessary computational cost or reconstruct incorrect results. On the other hand, they neglect the connections between the up-exposure and down-exposure models and thus fail to fully excavate effective features. In this paper, we revisit the stack-based ITM approaches and propose a novel method to reconstruct HDR radiance from a single image, which only needs to estimate two exposure images. At first, we design the exposure adaptive block that can adaptively adjust the exposure based on the luminance distribution of the input image. Secondly, we devise the cross-model attention block to connect the exposure adjustment models. Thirdly, we propose an end-to-end ITM pipeline by incorporating the multi-exposure fusion model. Furthermore, we propose and open a multi-exposure dataset that indicates the optimal exposure-up/down levels. Experimental results show that the proposed method outperforms some state-of-the-art methods.

        ----

        ## [874] Combining Implicit-Explicit View Correlation for Light Field Semantic Segmentation

        **Authors**: *Ruixuan Cong, Da Yang, Rongshan Chen, Sizhe Wang, Zhenglong Cui, Hao Sheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00885](https://doi.org/10.1109/CVPR52729.2023.00885)

        **Abstract**:

        Since light field simultaneously records spatial information and angular information of light rays, it is considered to be beneficial for many potential applications, and semantic segmentation is one of them. The regular variation of image information across views facilitates a comprehensive scene understanding. However, in the case of limited memory, the high-dimensional property of light field makes the problem more intractable than generic semantic segmentation, manifested in the difficulty of fully exploiting the relationships among views while maintaining contextual information in single view. In this paper, we propose a novel network called LF-IENet for light field semantic segmentation. It contains two different manners to mine complementary information from surrounding views to segment central view. One is implicit feature integration that leverages attention mechanism to compute inter-view and intra-view similarity to modulate features of central view. The other is explicit feature propagation that directly warps features of other views to central view under the guidance of disparity. They complement each other and jointly realize complementary information fusion across views in light field. The proposed method achieves outperforming performance on both real-world and synthetic light field datasets, demonstrating the effectiveness of this new architecture.

        ----

        ## [875] 3D Spatial Multimodal Knowledge Accumulation for Scene Graph Prediction in Point Cloud

        **Authors**: *Mingtao Feng, Haoran Hou, Liang Zhang, Zijie Wu, Yulan Guo, Ajmal Mian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00886](https://doi.org/10.1109/CVPR52729.2023.00886)

        **Abstract**:

        In-depth understanding of a 3D scene not only involves locating/recognizing individual objects, but also requires to infer the relationships and interactions among them. However, since 3D scenes contain partially scanned objects with physical connections, dense placement, changing sizes, and a wide variety of challenging relationships, existing methods perform quite poorly with limited training samples. In this work, we find that the inherently hierarchical structures of physical space in 3D scenes aid in the automatic association of semantic and spatial arrangements, specifying clear patterns and leading to less ambiguous predictions. Thus, they well meet the challenges due to the rich variations within scene categories. To achieve this, we explicitly unify these structural cues of 3D physical spaces into deep neural networks to facilitate scene graph prediction. Specifically, we exploit an external knowledge base as a baseline to accumulate both contextualized visual content and textual facts to form a 3D spatial multimodal knowledge graph. Moreover, we propose a knowledge-enabled scene graph prediction module benefiting from the 3D spatial knowledge to effectively regularize semantic space of relationships. Extensive experiments demonstrate the superiority of the proposed method over current state-of-the-art competitors. Our code is available at https://github.com/HHrEtvP/SMKA.

        ----

        ## [876] Role of Transients in Two-Bounce Non-Line-of-Sight Imaging

        **Authors**: *Siddharth Somasundaram, Akshat Dave, Connor Henley, Ashok Veeraraghavan, Ramesh Raskar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00887](https://doi.org/10.1109/CVPR52729.2023.00887)

        **Abstract**:

        The goal of non-line-of-sight (NLOS) imaging is to image objects occluded from the camera's field of view using multiply scattered light. Recent works have demonstrated the feasibility of two-bounce (2B) NLOS imaging by scanning a laser and measuring cast shadows of occluded objects in scenes with two relay surfaces. In this work, we study the role of time-of-flight (ToF) measurements, i.e. transients, in 2B-NLOS under multiplexed illumination. Specifically, we study how ToF information can reduce the number of measurements and spatial resolution needed for shape reconstruction. We present our findings with respect to tradeoffs in (1) temporal resolution, (2) spatial resolution, and (3) number of image captures by studying SNR and recoverability as functions of system parameters. This leads to a formal definition of the mathematical constraints for 2B lidar. We believe that our work lays an analytical ground- workfor design of future NLOS imaging systems, especially as ToF sensors become increasingly ubiquitous.

        ----

        ## [877] 3D Concept Learning and Reasoning from Multi-View Images

        **Authors**: *Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B. Tenenbaum, Chuang Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00888](https://doi.org/10.1109/CVPR52729.2023.00888)

        **Abstract**:

        Humans are able to accurately reason in 3D by gathering multi-view observations of the surrounding world. Inspired by this insight, we introduce a new large-scale benchmark for 3D multi-view visual question answering (3DMV-VQA). This dataset is collected by an embodied agent actively moving and capturing RGB images in an environment using the Habitat simulator. In total, it consists of approximately 5k scenes, 600k images, paired with 50k questions. We evaluate various state-of-the-art models for visual reasoning on our benchmark and find that they all perform poorly. We suggest that a principled approach for 3D reasoning from multi-view images should be to infer a compact 3D representation of the world from the multi-view images, which is further grounded on open-vocabulary semantic concepts, and then to execute reasoning on these 3D representations. As the first step towards this approach, we propose a novel 3D concept learning and reasoning (3D-CLR) framework that seamlessly combines these components via neural fields, 2D pre-trained vision-language models, and neural reasoning operators. Experimental results suggest that our framework outperforms baseline models by a large margin, but the challenge remains largely unsolved. We further perform an in-depth analysis of the challenges and highlight potential future directions..

        ----

        ## [878] Viewpoint Equivariance for Multi-View 3D Object Detection

        **Authors**: *Dian Chen, Jie Li, Vitor Guizilini, Rares Ambrus, Adrien Gaidon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00889](https://doi.org/10.1109/CVPR52729.2023.00889)

        **Abstract**:

        3D object detection from visual sensors is a corner-stone capability of robotic systems. State-of-the-art methods focus on reasoning and decoding object bounding boxes from multi-view camera input. In this work we gain intuition from the integral role of multi-view consistency in 3D scene understanding and geometric learning. To this end, we introduce VEDet, a novel 3D object detection framework that exploits 3D multi-view geometry to improve localization through viewpoint awareness and equivariance. VEDet leverages a query-based transformer architecture and encodes the 3D scene by augmenting image features with positional encodings from their 3D perspective geometry. We design view-conditioned queries at the output level, which enables the generation of multiple virtual frames during training to learn viewpoint equivariance by enforcing multi-view consistency. The multi-view geometry injected at the input level as positional encodings and regularized at the loss level provides rich geometric cues for 3D object detection, leading to state-of-the-art performance on the nuScenes benchmark. The code and model are made available at https://github.com/TRI-ML/VEDet.

        ----

        ## [879] Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction

        **Authors**: *Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, Jiwen Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00890](https://doi.org/10.1109/CVPR52729.2023.00890)

        **Abstract**:

        Modern methods for vision-centric autonomous driving perception widely adopt the bird's-eye-view (BEV) representation to describe a 3D scene. Despite its better efficiency than voxel representation, it has difficulty describing the fine-grained 3D structure of a scene with a single plane. To address this, we propose a tri-perspective view (TPV) representation which accompanies BEV with two additional perpendicular planes. We model each point in the 3D space by summing its projected features on the three planes. To lift image features to the 3D TPV space, we further propose a transformer-based TPV encoder (TPVFormer) to obtain the TPV features effectively. We employ the attention mechanism to aggregate the image features corresponding to each query in each TPV plane. Experiments show that our model trained with sparse supervision effectively predicts the semantic occupancy for all voxels. We demonstrate for the first time that using only camera inputs can achieve comparable performance with LiDAR-based methods on the LiDAR segmentation task on nuScenes. Code: https://github.com/wzzheng/TPVFormer.

        ----

        ## [880] BEV@DC: Bird's-Eye View Assisted Training for Depth Completion

        **Authors**: *Wending Zhou, Xu Yan, Yinghong Liao, Yuankai Lin, Jin Huang, Gangming Zhao, Shuguang Cui, Zhen Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00891](https://doi.org/10.1109/CVPR52729.2023.00891)

        **Abstract**:

        Depth completion plays a crucial role in autonomous driving, in which cameras and LiDARs are two complementary sensors. Recent approaches attempt to exploit spatial geometric constraints hidden in LiDARs to enhance image-guided depth completion. However, only low efficiency and poor generalization can be achieved. In this paper, we propose BEV@DC, a more efficient and powerful multi-modal training scheme, to boost the performance of image-guided depth completion. In practice, the proposed BEV@DC model comprehensively takes advantage of LiDARs with rich geometric details in training, employing an enhanced depth completion manner in inference, which takes only images (RGB and depth) as input. Specifically, the geometric-aware LiDAR features are projected onto a unified BEV space, combining with RGB features to perform BEV completion. By equipping a newly proposed point-voxel spatial propagation network (PV-SPN), this auxiliary branch introduces strong guidance to the original image branches via 3D dense supervision and feature consistency. As a result, our baseline model demonstrates significant improvements with the sole image inputs. Concretely, it achieves state-of-the-art on several benchmarks, e.g., ranking Top-1 on the challenging KITTI depth completion benchmark.

        ----

        ## [881] Collaboration Helps Camera Overtake LiDAR in 3D Detection

        **Authors**: *Yue Hu, Yifan Lu, Runsheng Xu, Weidi Xie, Siheng Chen, Yanfeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00892](https://doi.org/10.1109/CVPR52729.2023.00892)

        **Abstract**:

        Camera-only 3D detection provides an economical solution with a simple configuration for localizing objects in 3D space compared to LiDAR-based detection systems. However, a major challenge lies in precise depth estimation due to the lack of direct 3D measurements in the input. Many previous methods attempt to improve depth estimation through network designs, e.g., deformable layers and larger receptive fields. This work proposes an orthogonal direction, improving the camera-only 3D detection by introducing multi-agent collaborations. Our proposed collaborative camera-only 3D detection (CoCa3D) enables agents to share complementary information with each other through communication. Meanwhile, we optimize communication efficiency by selecting the most informative cues. The shared messages from multiple view-points disambiguate the single-agent estimated depth and complement the occluded and long-range regions in the single-agent view. We evaluate CoCa3D in one real-world dataset and two new simulation datasets. Results show that CoCa3D improves previous SOTA performances by 44.21% on DAIR-V2X, 30.60% on OPV2V+, 12.59% on CoPerception-UAVs+ for AP@70. Our preliminary results show a potential that with sufficient collaboration, the camera might overtake LiDAR in some practical scenarios. We released the dataset and code.

        ----

        ## [882] Uni3D: A Unified Baseline for Multi-Dataset 3D Object Detection

        **Authors**: *Bo Zhang, Jiakang Yuan, Botian Shi, Tao Chen, Yikang Li, Yu Qiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00893](https://doi.org/10.1109/CVPR52729.2023.00893)

        **Abstract**:

        Current 3D object detection models follow a single dataset-specific training and testing paradigm, which often faces a serious detection accuracy drop when they are directly deployed in another dataset. In this paper, we study the task of training a unified 3D detector from multiple datasets. We observe that this appears to be a challenging task, which is mainly due to that these datasets present substantial data-level differences and taxonomy-level variations caused by different LiDAR types and data acquisition standards. Inspired by such observation, we present a Uni3D which leverages a simple data-level correction operation and a designed semantic-level coupling-and-recoupling module to alleviate the unavoidable data-level and taxonomy-level differences, respectively. Our method is simple and easily combined with many 3D object detection baselines such as PV-RCNN and Voxel-RCNN, enabling them to effectively learn from multiple off-the-shelf 3D datasets to obtain more discriminative and generalizable representations. Experiments are conducted on many dataset consolidation settings. Their results demonstrate that Uni3D exceeds a series of individual detectors trained on a single dataset, with a 1.04× parameter increase over a selected baseline detector. We expect this work will inspire the research of 3D generalization since it will push the limits of perceptual performance. Our code is available at: https://github.com/PJLab-ADG/3DTrans.

        ----

        ## [883] Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration

        **Authors**: *Kemal Oksuz, Tom Joy, Puneet K. Dokania*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00894](https://doi.org/10.1109/CVPR52729.2023.00894)

        **Abstract**:

        The current approach for testing the robustness of object detectors suffers from serious deficiencies such as improper methods of performing out-of-distribution detection and using calibration metrics which do not consider both localisation and classification quality. In this work, we address these issues, and introduce the Self Aware Object Detection (SAOD) task, a unified testing framework which respects and adheres to the challenges that object detectors face in safety-critical environments such as autonomous driving. Specifically, the SAOD task requires an object detector to be: robust to domain shift; obtain reliable uncertainty estimates for the entire scene; and provide calibrated confidence scores for the detections. We extensively use our framework, which introduces novel metrics and large scale test datasets, to test numerous object detectors in two different use-cases, allowing us to highlight critical insights into their robustness performance. Finally, we introduce a simple baseline for the SAOD task, enabling researchers to benchmark future proposed methods and move towards robust object detectors which are fit for purpose. Code is available at: https://github.com/fiveai/saod.

        ----

        ## [884] Depth Estimation from Camera Image and mmWave Radar Point Cloud

        **Authors**: *Akash Deep Singh, Yunhao Ba, Ankur Sarker, Howard Zhang, Achuta Kadambi, Stefano Soatto, Mani B. Srivastava, Alex Wong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00895](https://doi.org/10.1109/CVPR52729.2023.00895)

        **Abstract**:

        We present a method for inferring dense depth from a camera image and a sparse noisy radar point cloud. We first describe the mechanics behind mmWave radar point cloud formation and the challenges that it poses, i.e. ambiguous elevation and noisy depth and azimuth components that yields incorrect positions when projected onto the image, and how existing works have overlooked these nuances in camera-radar fusion. Our approach is motivated by these mechanics, leading to the design of a network that maps each radar point to the possible surfaces that it may project onto in the image plane. Unlike existing works, we do not process the raw radar point cloud as an erroneous depth map, but query each raw point independently to associate it with likely pixels in the image – yielding a semi-dense radar depth map. To fuse radar depth with an image, we propose a gated fusion scheme that accounts for the confidence scores of the correspondence so that we selectively combine radar and camera embeddings to yield a dense depth map. We test our method on the NuScenes benchmark and show a 10.3% improvement in mean absolute error and a 9.1% improvement in root-mean-square error over the best method. Code: https://github.com/nesl/radar-camera-fusion-depth.

        ----

        ## [885] SGLoc: Scene Geometry Encoding for Outdoor LiDAR Localization

        **Authors**: *Wen Li, Shangshu Yu, Cheng Wang, Guosheng Hu, Siqi Shen, Chenglu Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00896](https://doi.org/10.1109/CVPR52729.2023.00896)

        **Abstract**:

        LiDAR-based absolute pose regression estimates the global pose through a deep network in an end-to-end manner, achieving impressive results in learning-based localization. However, the accuracy of existing methods still has room to improve due to the difficulty of effectively encoding the scene geometry and the unsatisfactory quality of the data. In this work, we propose a novel LiDAR localization frame-work, SGLoc, which decouples the pose estimation to point cloud correspondence regression and pose estimation via this correspondence. This decoupling effectively encodes the scene geometry because the decoupled correspondence regression step greatly preserves the scene geometry, leading to significant performance improvement. Apart from this decoupling, we also design a tri-scale spatial feature aggregation module and inter-geometric consistency constraint loss to effectively capture scene geometry. Moreover, we empirically find that the ground truth might be noisy due to GPS/INS measuring errors, greatly reducing the pose estimation performance. Thus, we propose a pose quality evaluation and enhancement method to measure and correct the ground truth pose. Extensive experiments on the Oxford Radar RobotCar and NCLT datasets demonstrate the effectiveness of SGLoc, which outperforms state-of-the-art regression-based localization methods by 68.5% and 67.6% on position accuracy, respectively.

        ----

        ## [886] ConQueR: Query Contrast Voxel-DETR for 3D Object Detection

        **Authors**: *Benjin Zhu, Zhe Wang, Shaoshuai Shi, Hang Xu, Lanqing Hong, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00897](https://doi.org/10.1109/CVPR52729.2023.00897)

        **Abstract**:

        Although DETR-based 3D detectors simplify the detection pipeline and achieve direct sparse predictions, their performance still lags behind dense detectors with post-processing for 3D object detection from point clouds. DETRs usually adopt a larger number of queries than GTs (e.g., 300 queries v.s. ~40 objects in Waymo) in a scene, which inevitably incur many false positives during inference. In this paper, we propose a simple yet effective sparse 3D detector, named Query Contrast Voxel-DETR (Con-QueR), to eliminate the challenging false positives, and achieve more accurate and sparser predictions. We observe that most false positives are highly overlapping in local regions, caused by the lack of explicit supervision to discriminate locally similar queries. We thus propose a Query Contrast mechanism to explicitly enhance queries towards their best-matched GTs over all unmatched query predictions. This is achieved by the construction of positive and negative GT-query pairs for each GT, and a contrastive loss to enhance positive GT-query pairs against negative ones based on feature similarities. ConQueR closes the gap of sparse and dense 3D detectors, and reduces ~60% false positives. Our single-frame ConQueR achieves 71.6 mAPH/L2 on the challenging Waymo Open Dataset validation set, outper-forming previous sota methods by over 2.0 mAPH/L2. Code

        ----

        ## [887] DeepMapping2: Self-Supervised Large-Scale LiDAR Map Optimization

        **Authors**: *Chao Chen, Xinhao Liu, Yiming Li, Li Ding, Chen Feng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00898](https://doi.org/10.1109/CVPR52729.2023.00898)

        **Abstract**:

        LiDAR mapping is important yet challenging in selfdriving and mobile robotics. To tackle such a global point cloud registration problem, DeepMapping [1] converts the complex map estimation into a self-supervised training of simple deep networks. Despite its broad convergence range on small datasets, DeepMapping still cannot produce satisfactory results on large-scale datasets with thousands of frames. This is due to the lack of loop closures and exact cross-frame point correspondences, and the slow convergence of its global localization network. We propose DeepMapping2 by adding two novel techniques to address these issues: (1) organization of training batch based on map topology from loop closing, and (2) self-supervised local-to-global point consistency loss leveraging pairwise registration. Our experiments and ablation studies on public datasets such as KITTI, NCLT, and Nebula demonstrate the effectiveness of our method.

        ----

        ## [888] Towards Unsupervised Object Detection from LiDAR Point Clouds

        **Authors**: *Lunjun Zhang, Anqi Joyce Yang, Yuwen Xiong, Sergio Casas, Bin Yang, Mengye Ren, Raquel Urtasun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00899](https://doi.org/10.1109/CVPR52729.2023.00899)

        **Abstract**:

        In this paper, we study the problem of unsupervised object detection from 3D point clouds in self-driving scenes. We present a simple yet effective method that exploits (i) point clustering in near-range areas where the point clouds are dense, (ii) temporal consistency to filter out noisy unsupervised detections, (iii) translation equivariance of CNNs to extend the auto-labels to long range, and (iv) self-supervision for improving on its own. Our approach, OYSTER (Object Discovery via Spatio-Temporal Refinement), does not impose constraints on data collection (such as repeated traversals of the same location), is able to detect objects in a zero-shot manner without supervised fine-tuning (even in sparse, distant regions), and continues to self-improve given more rounds of iterative self-training. To better measure model performance in self-driving scenarios, we propose a new planning-centric perception metric based on distance-to-collision. We demonstrate that our unsupervised object detector significantly outperforms unsupervised baselines on PandaSet and Argoverse 2 Sensor dataset, showing promise that self-supervision combined with object priors can enable object discovery in the wild. For more information, visit the project website: https://waabi.ai/research/oyster.

        ----

        ## [889] MoDAR: Using Motion Forecasting for 3D Object Detection in Point Cloud Sequences

        **Authors**: *Yingwei Li, Charles R. Qi, Yin Zhou, Chenxi Liu, Dragomir Anguelov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00900](https://doi.org/10.1109/CVPR52729.2023.00900)

        **Abstract**:

        Occluded and long-range objects are ubiquitous and challenging for 3D object detection. Point cloud sequence data provide unique opportunities to improve such cases, as an occluded or distant object can be observed from different viewpoints or gets better visibility over time. However, the efficiency and effectiveness in encoding longterm sequence data can still be improved. In this work, we propose MoDAR, using motion forecasting outputs as a type of virtual modality, to augment LiDAR point clouds. The MoDAR modality propagates object information from temporal contexts to a target frame, represented as a set of virtual points, one for each object from a waypoint on a forecasted trajectory. A fused point cloud of both raw sensor points and the virtual points can then be fed to any off-the-shelf point-cloud based 3D object detector. Evaluated on the Waymo Open Dataset, our method significantly improves prior art detectors by using motion forecasting from extra-long sequences (e.g. 18 seconds), achieving new state of the arts, while not adding much computation overhead.

        ----

        ## [890] Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision

        **Authors**: *Fangqiang Ding, Andras Palffy, Dariu M. Gavrila, Chris Xiaoxuan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00901](https://doi.org/10.1109/CVPR52729.2023.00901)

        **Abstract**:

        This work proposes a novel approach to 4D radar-based scene flow estimation via cross-modal learning. Our approach is motivated by the co-located sensing redundancy in modern autonomous vehicles. Such redundancy implicitly provides various forms of supervision cues to the radar scene flow estimation. Specifically, we introduce a multi-task model architecture for the identified cross-modal learning problem and propose loss functions to opportunistically engage scene flow estimation using multiple cross-modal constraints for effective model training. Extensive experiments show the state-of-the-art performance of our method and demonstrate the effectiveness of cross-modal super-vised learning to infer more accurate 4D radar scene flow. We also show its usefulness to two subtasks - motion segmentation and ego-motion estimation. Our source code will be available on https://github.com/Toytiny/CMFlow.

        ----

        ## [891] Instant Domain Augmentation for LiDAR Semantic Segmentation

        **Authors**: *Kwonyoung Ryu, Soonmin Hwang, Jaesik Park*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00902](https://doi.org/10.1109/CVPR52729.2023.00902)

        **Abstract**:

        Despite the increasing popularity of LiDAR sensors, perception algorithms using 3D LiDAR data struggle with the sensor-bias problem. Specifically, the performance of perception algorithms significantly drops when an unseen specification of the LiDAR sensor is applied at test time due to the domain discrepancy. This paper presents a fast and flexible LiDAR augmentation method for the semantic segmentation task called LiDomAug. It aggregates raw LiDAR scans and creates a LiDAR scan of any configurations with the consideration of dynamic distortion and occlusion, resulting in instant domain augmentation. Our on-demand augmentation module runs at 330 FPS, so it can be seamlessly integrated into the data loader in the learning framework. In our experiments, learning-based approaches aided with the proposed LiDomAug are less affected by the sensor-bias issue and achieve new state-of-the-art domain adaptation performances on SemanticKITTI and nuScenes dataset without the use of the target domain data. We also present a sensor-agnostic model that faithfully works on the various LiDAR configurations.

        ----

        ## [892] Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation

        **Authors**: *Li Li, Hubert P. H. Shum, Toby P. Breckon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00903](https://doi.org/10.1109/CVPR52729.2023.00903)

        **Abstract**:

        Whilst the availability of 3D LiDAR point cloud data has significantly grown in recent years, annotation remains expensive and time-consuming, leading to a demand for semi-supervised semantic segmentation methods with application domains such as autonomous driving. Existing work very often employs relatively large segmentation backbone networks to improve segmentation accuracy, at the expense of computational costs. In addition, many use uniform sampling to reduce ground truth data requirements for learning needed, often resulting in sub-optimal performance. To address these issues, we propose a new pipeline that employs a smaller architecture, requiring fewer ground-truth annotations to achieve superior segmentation accuracy compared to contemporary approaches. This is facilitated via a novel Sparse Depthwise Separable Convolution module that significantly reduces the network parameter count while retaining overall task performance. To effectively sub-sample our training data, we propose a new Spatio-Temporal Redundant Frame Downsampling (ST-RFD) method that leverages knowledge of sensor motion within the environment to extract a more diverse subset of training data frame samples. To leverage the use of limited annotated data samples, we further propose a soft pseudo-label method informed by Li-DAR reflectivity. Our method outperforms contemporary semi-supervised work in terms of mIoU, using less labeled data, on the SemanticKITTI (59.5@5%) and ScribbleKITTI (58.1@5%) benchmark datasets, based on a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$2.3\times reduction$</tex>
 in model parameters and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$64l\times fewer$</tex>
 multiply-add operations whilst also demonstrating significant performance improvement on limited training data (i.e., Less is More).

        ----

        ## [893] MarS3D: A Plug-and-Play Motion-Aware Model for Semantic Segmentation on Multi-Scan 3D Point Clouds

        **Authors**: *Jiahui Liu, Chirui Chang, Jianhui Liu, Xiaoyang Wu, Lan Ma, Xiaojuan Qi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00904](https://doi.org/10.1109/CVPR52729.2023.00904)

        **Abstract**:

        3D semantic segmentation on multi-scan large-scale point clouds plays an important role in autonomous systems. Unlike the single-scan-based semantic segmentation task, this task requires distinguishing the motion states of points in addition to their semantic categories. However, methods designed for single-scan-based segmentation tasks perform poorly on the multi-scan task due to the lacking of an effective way to integrate temporal information. We propose MarS3D, a plug-and-play motion-aware module for semantic segmentation on multi-scan 3D point clouds. This module can be flexibly combined with single-scan models to allow them to have multi-scan perception abilities. The model encompasses two key designs: the Cross-Frame Feature Embedding module for enriching representation learning and the Motion-Aware Feature Learning module for enhancing motion awareness. Extensive experiments show that MarS3D can improve the performance of the baseline model by a large margin. The code is available at https://github.com/CVMI-Lab/MarS3D.

        ----

        ## [894] 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds

        **Authors**: *Aoran Xiao, Jiaxing Huang, Weihao Xuan, Ruijie Ren, Kangcheng Liu, Dayan Guan, Abdulmotaleb El-Saddik, Shijian Lu, Eric P. Xing*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00905](https://doi.org/10.1109/CVPR52729.2023.00905)

        **Abstract**:

        Robust point cloud parsing under all-weather conditions is crucial to level-5 autonomy in autonomous driving. However, how to learn a universal 3D semantic segmentation (3DSS) model is largely neglected as most existing benchmarks are dominated by point clouds captured under normal weather. We introduce SemanticSTF, an adverse-weather point cloud dataset that provides dense point-level annotations and allows to study 3DSS under various adverse weather conditions. We study all-weather 3DSS modeling under two setups: 1) domain adaptive 3DSS that adapts from normal-weather data to adverse-weather data; 2) domain generalizable 3DSS that learns all-weather 3DSS models from normal-weather data. Our studies reveal the challenge while existing 3DSS methods encounter adverse-weather data, showing the great value of SemanticSTF in steering the future endeavor along this very meaningful research direction. In addition, we design a domain randomization technique that alternatively randomizes the geometry styles of point clouds and aggregates their embeddings, ultimately leading to a generalizable model that can improve 3DSS under various adverse weather effectively. The SemanticSTF and related codes are available at https://github.com/xiaoaoran/SemanticSTF.

        ----

        ## [895] Novel Class Discovery for 3D Point Cloud Semantic Segmentation

        **Authors**: *Luigi Riz, Cristiano Saltori, Elisa Ricci, Fabio Poiesi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00906](https://doi.org/10.1109/CVPR52729.2023.00906)

        **Abstract**:

        Novel class discovery (NCD) for semantic segmentation is the task of learning a model that can segment unlabelled (novel) classes using only the supervision from labelled (base) classes. This problem has recently been pioneered for 2D image data, but no work exists for 3D point cloud data. In fact, the assumptions made for 2D are loosely applicable to 3D in this case. This paper is presented to advance the state of the art on point cloud data analysis in four directions. Firstly, we address the new problem of NCD for point cloud semantic segmentation. Secondly, we show that the transposition of the only existing NCD method for 2D semantic segmentation to 3D data is sub-optimal. Thirdly, we present a new method for NCD based on online clustering that exploits uncertainty quantification to produce prototypes for pseudo-labelling the points of the novel classes. Lastly, we introduce a new evaluation protocol to assess the performance of NCD for point cloud semantic segmentation. We thoroughly evaluate our method on SemanticKITTI and SemanticPOSS datasets, showing that it can significantly outperform the baseline. Project page: https://github.com/LuigiRiz/NOPS.

        ----

        ## [896] GD-MAE: Generative Decoder for MAE Pre-Training on LiDAR Point Clouds

        **Authors**: *Honghui Yang, Tong He, Jiaheng Liu, Hua Chen, Boxi Wu, Binbin Lin, Xiaofei He, Wanli Ouyang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00907](https://doi.org/10.1109/CVPR52729.2023.00907)

        **Abstract**:

        Despite the tremendous progress of Masked Autoencoders (MAE) in developing vision tasks such as image and video, exploring MAE in large-scale 3D point clouds remains challenging due to the inherent irregularity. In contrast to previous 3D MAE frameworks, which either design a complex decoder to infer masked information from maintained regions or adopt sophisticated masking strategies, we instead propose a much simpler paradigm. The core idea is to apply a Generative Decoder for MAE (GD-MAE) to automatically merges the surrounding context to restore the masked geometric knowledge in a hierarchical fusion manner. In doing so, our approach is free from introducing the heuristic design of decoders and enjoys the flexibility of exploring various masking strategies. The corresponding part costs less than 12% latency compared with conventional methods, while achieving better performance. We demonstrate the efficacy of the proposed method on several large-scale benchmarks: Waymo, KITTI, and ONCE. Consistent improvement on downstream detection tasks illustrates strong robustness and generalization capability. Not only our method reveals state-of-the-art results, but remarkably, we achieve comparable accuracy even with 20% of the labeled data on the Waymo dataset. Code will be released.

        ----

        ## [897] Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning

        **Authors**: *Xiaoyang Wu, Xin Wen, Xihui Liu, Hengshuang Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00908](https://doi.org/10.1109/CVPR52729.2023.00908)

        **Abstract**:

        As a pioneering work, PointContrast conducts unsupervised 3D representation learning via leveraging contrastive learning over raw RGB-D frames and proves its effectiveness on various downstream tasks. However, the trend of large-scale unsupervised learning in 3D has yet to emerge due to two stumbling blocks: the inefficiency of matching RGB-D frames as contrastive views and the annoying mode collapse phenomenon mentioned in previous works. Turning the two stumbling blocks into empirical stepping stones, we first propose an efficient and effective contrastive learning framework, which generates contrastive views directly on scene-level point clouds by a well-curated data augmentation pipeline and a practical view mixing strategy. Second, we introduce reconstructive learning on the contrastive learning framework with an exquisite design of contrastive cross masks, which targets the reconstruction of point color and surfel normal. Our Masked Scene Contrast (MSC) framework is capable of extracting comprehensive 3D representations more efficiently and effectively. It accelerates the pretraining procedure by at least 3 x and still achieves an uncompromised performance compared with previous work. Besides, MSC also enables large-scale 3D pre-training across multiple datasets, which further boosts the performance and achieves state-of-the-art fine-tuning results on several downstream tasks, e.g., 75.5% mIoU on ScanNet semantic segmentation validation set.

        ----

        ## [898] Open-set Semantic Segmentation for Point Clouds via Adversarial Prototype Framework

        **Authors**: *Jianan Li, Qiulei Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00909](https://doi.org/10.1109/CVPR52729.2023.00909)

        **Abstract**:

        Recently, point cloud semantic segmentation has attracted much attention in computer vision. Most of the existing works in literature assume that the training and testing point clouds have the same object classes, but they are generally invalid in many real-world scenarios for identifying the 3D objects whose classes are not seen in the training set. To address this problem, we propose an Adversarial Prototype Framework (APF) for handling the open-set 3D semantic segmentation task, which aims to identify 3D unseen-class points while maintaining the segmentation performance on seen-class points. The proposed APF consists of a feature extraction module for extracting point features, a prototypical constraint module, and a feature adversarial module. The prototypical constraint module is designed to learn prototypes for each seen class from point features. The feature adversarial module utilizes generative adversarial networks to estimate the distribution of unseenclass features implicitly, and the synthetic unseen-class features are utilized to prompt the model to learn more effective point features and prototypes for discriminating unseen-class samples from the seen-class ones. Experimental results on two public datasets demonstrate that the proposed APF outperforms the comparative methods by a large margin in most cases.

        ----

        ## [899] ACL-SPC: Adaptive Closed-Loop System for Self-Supervised Point Cloud Completion

        **Authors**: *Sangmin Hong, Mohsen Yavartanoo, Reyhaneh Neshatavar, Kyoung Mu Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00910](https://doi.org/10.1109/CVPR52729.2023.00910)

        **Abstract**:

        Point cloud completion addresses filling in the missing parts of a partial point cloud obtained from depth sensors and generating a complete point cloud. Although there has been steep progress in the supervised methods on the synthetic point cloud completion task, it is hardly applicable in real-world scenarios due to the domain gap between the synthetic and real-world datasets or the requirement of prior information. To overcome these limitations, we propose a novel self-supervised framework ACL-SPC for point cloud completion to train and test on the same data. ACL-SPC takes a single partial input and attempts to output the complete point cloud using an adaptive closed-loop (ACL) system that enforces the output same for the variation of an input. We evaluate our ACL-SPC on various datasets to prove that it can successfully learn to complete a partial point cloud as the first self-supervised scheme. Results show that our method is comparable with unsupervised methods and achieves superior performance on the real-world dataset compared to the supervised methods trained on the synthetic dataset. Extensive experiments justify the necessity of self-supervised learning and the effectiveness of our proposed method for the real-world point cloud completion task. The code is publicly available from this link.

        ----

        ## [900] Fast Point Cloud Generation with Straight Flows

        **Authors**: *Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan, Raghuraman Krishnamoorthi, Vikas Chandra, Qiang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00911](https://doi.org/10.1109/CVPR52729.2023.00911)

        **Abstract**:

        Diffusion models have emerged as a powerful tool for point cloud generation. A key component that drives the impressive performance for generating high-quality samples from noise is iteratively denoise for thousands of steps. While beneficial, the complexity of learning steps has limited its applications to many 3D real-world. To address this limitation, we propose Point Straight Flow (PSF), a model that exhibits impressive performance using one step. Our idea is based on the reformulation of the standard diffusion model, which optimizes the curvy learning trajectory into a straight path. Further, we develop a distillation strategy to shorten the straight path into one step without a performance loss, enabling applications to 3D real-world with latency constraints. We perform evaluations on multiple 3D tasks and find that our PSF performs comparably to the standard diffusion model, outperforming other efficient 3D point cloud generation methods. On real-world applications such as point cloud completion and training-free text-guided generation in a low-latency setup, PSF performs favorably.

        ----

        ## [901] PointVector: A Vector Representation In Point Cloud Analysis

        **Authors**: *Xin Deng, Wenyu Zhang, Qing Ding, Xinming Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00912](https://doi.org/10.1109/CVPR52729.2023.00912)

        **Abstract**:

        In point cloud analysis, point-based methods have rapidly developed in recent years. These methods have recently focused on concise MLP structures, such as Point-NeXt, which have demonstrated competitiveness with Convolutional and Transformer structures. However, standard MLPs are limited in their ability to extract local features effectively. To address this limitation, we propose a Vector-oriented Point Set Abstraction that can aggregate neighboring features through higher-dimensional vectors. To facilitate network optimization, we construct a transformation from scalar to vector using independent angles based on 3D vector rotations. Finally, we develop a PointVector model that follows the structure of PointNeXt. Our experimental results demonstrate that PointVector achieves state-of-the-art performance 72.3% mIOU on the S3DIS Area 5 and 78.4% mIOU on the S3DIS (6-fold cross-validation) with only 58% model parameters of PointNeXt. We hope our work will help the exploration of concise and effective feature representations. The code will be released soon.

        ----

        ## [902] ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer

        **Authors**: *Shanshan Li, Pan Gao, Xiaoyang Tan, Mingqiang Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00913](https://doi.org/10.1109/CVPR52729.2023.00913)

        **Abstract**:

        Problems such as equipment defects or limited view-points will lead the captured point clouds to be incomplete. Therefore, recovering the complete point clouds from the partial ones plays an vital role in many practical tasks, and one of the keys lies in the prediction of the missing part. In this paper, we propose a novel point cloud completion approach namely ProxyFormer that divides point clouds into existing (input) and missing (to be predicted) parts and each part communicates information through its proxies. Specifically, we fuse information into point proxy via feature and position extractor, and generate features for missing point proxies from the features of existing point proxies. Then, in order to better perceive the position of missing points, we design a missing part sensitive transformer, which converts random normal distribution into reasonable position information, and uses proxy alignment to refine the missing proxies. It makes the predicted point proxies more sensitive to the features and positions of the missing part, and thus makes these proxies more suitable for subsequent coarse-to-fine processes. Experimental results show that our method outperforms state-of-the-art completion networks on several benchmark datasets and has the fastest inference speed.

        ----

        ## [903] FAC: 3D Representation Learning via Foreground Aware Feature Contrast

        **Authors**: *Kangcheng Liu, Aoran Xiao, Xiaoqin Zhang, Shijian Lu, Ling Shao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00914](https://doi.org/10.1109/CVPR52729.2023.00914)

        **Abstract**:

        Contrastive learning has recently demonstrated great potential for unsupervised pre-training in 3D scene understanding tasks. However, most existing work randomly selects point features as anchors while building contrast, leading to a clear bias toward background points that often dominate in 3D scenes. Also, object awareness and foreground-to-background discrimination are neglected, making contrastive learning less effective. To tackle these issues, we propose a general foreground-aware feature contrast (FAC) framework to learn more effective point cloud representations in pre-training. FAC consists of two novel contrast designs to construct more effective and informative contrast pairs. The first is building positive pairs within the same foreground segment where points tend to have the same semantics. The second is that we prevent over-discrimination between 3D segments/objects and encourage foreground-to-background distinctions at the segment level with adaptive feature learning in a Siamese correspondence network, which adaptively learns feature correlations within and across point cloud views effectively. Visualization with point activation maps shows that our contrast pairs capture clear correspondences among fore-ground regions during pre-training. Quantitative experiments also show that FAC achieves superior knowledge transfer and data efficiency in various downstream 3D semantic segmentation and object detection tasks. All codes, data, and models are available.

        ----

        ## [904] Rethinking the Approximation Error in 3D Surface Fitting for Point Cloud Normal Estimation

        **Authors**: *Hang Du, Xuejun Yan, Jingjing Wang, Di Xie, Shiliang Pu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00915](https://doi.org/10.1109/CVPR52729.2023.00915)

        **Abstract**:

        Most existing approaches for point cloud normal estimation aim to locally fit a geometric surface and calculate the normal from the fitted surface. Recently, learning-based methods have adopted a routine of predicting pointwise weights to solve the weighted least-squares surface fitting problem. Despite achieving remarkable progress, these methods overlook the approximation error of the fitting problem, resulting in a less accurate fitted surface. In this paper, we first carry out in-depth analysis of the approximation error in the surface fitting problem. Then, in order to bridge the gap between estimated and precise surface normals, we present two basic design principles: 1) applies the Z-direction Transform to rotate local patches for a better surface fitting with a lower approximation error; 2) models the error of the normal estimation as a learnable term. We implement these two principles using deep neural networks, and integrate them with the state-of-the-art (SOTA) normal estimation methods in a plug-and-play manner. Extensive experiments verify our approaches bring benefits to point cloud normal estimation and push the frontier of state-of-the-art performance on both synthetic and real-world datasets. The code is available at https://github.com/hikvision-research/3DVision.

        ----

        ## [905] PointCert: Point Cloud Classification with Deterministic Certified Robustness Guarantees

        **Authors**: *Jinghuai Zhang, Jinyuan Jia, Hongbin Liu, Neil Zhenqiang Gong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00916](https://doi.org/10.1109/CVPR52729.2023.00916)

        **Abstract**:

        Point cloud classification is an essential component in many security-critical applications such as autonomous driving and augmented reality. However, point cloud classifiers are vulnerable to adversarially perturbed point clouds. Existing certified defenses against adversarial point clouds suffer from a key limitation: their certified robustness guarantees are probabilistic, i.e., they produce an incorrect certified robustness guarantee with some probability. In this work, we propose a general framework, namely PointCert, that can transform an arbitrary point cloud classifier to be certifiably robust against adversarial point clouds with deterministic guarantees. PointCert certifiably predicts the same label for a point cloud when the number of arbitrarily added, deleted, and/or modified points is less than a threshold. Moreover, we propose multiple methods to optimize the certified robustness guarantees of PointCert in three application scenarios. We systematically evaluate PointCert on ModelNet and ScanObjectNN benchmark datasets. Our results show that PointCert substantially outperforms state-of-the-art certified defenses even though their robustness guarantees are probabilistic.

        ----

        ## [906] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting

        **Authors**: *Haiping Wang, Yuan Liu, Zhen Dong, Yulan Guo, Yu-Shen Liu, Wenping Wang, Bisheng Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00917](https://doi.org/10.1109/CVPR52729.2023.00917)

        **Abstract**:

        In this paper, we present a new method for the multi-view registration of point cloud. Previous multiview registration methods rely on exhaustive pairwise registration to construct a densely-connected pose graph and apply Iteratively Reweighted Least Square (IRLS) on the pose graph to compute the scan poses. However, constructing a densely-connected graph is time-consuming and contains lots of outlier edges, which makes the subsequent IRLS struggle to find correct poses. To address the above problems, we first propose to use a neural network to estimate the overlap between scan pairs, which enables us to construct a sparse but reliable pose graph. Then, we design a novel history reweighting function in the IRLS scheme, which has strong robustness to outlier edges on the graph. In comparison with existing multiview registration methods, our method achieves 11% higher registration recall on the 3DMatch dataset and ~ 13% lower registration errors on the ScanNet dataset while reducing ~ 70% required pairwise registrations. Comprehensive ablation studies are conducted to demonstrate the effectiveness of our designs. The source code is available at https://github.com/WHU-USI3DV/SGHR.

        ----

        ## [907] Visual Prompt Multi-Modal Tracking

        **Authors**: *Jiawen Zhu, Simiao Lai, Xin Chen, Dong Wang, Huchuan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00918](https://doi.org/10.1109/CVPR52729.2023.00918)

        **Abstract**:

        Visible-modal object tracking gives rise to a series of downstream multi-modal tracking tributaries. To inherit the powerful representations of the foundation model, a natural modus operandi for multi-modal tracking is full fine-tuning on the RGB-based parameters. Albeit effective, this manner is not optimal due to the scarcity of downstream data and poor transferability, etc. In this paper, inspired by the recent success of the prompt learning in language models, we develop Visual Prompt multi-modal Tracking (ViPT), which learns the modal-relevant prompts to adapt the frozen pre-trained foundation model to various downstream multi-modal tracking tasks. ViPT finds a better way to stimulate the knowledge of the RGB-based model that is pre-trained at scale, meanwhile only introducing a few trainable parameters (less than 1% of model parameters). ViPT outperforms the full fine-tuning paradigm on multiple downstream tracking tasks including RGB+Depth, RGB+Thermal, and RGB+Event tracking. Extensive experiments show the potential of visual prompt learning for multi-modal tracking, and ViPT can achieve state-of-the-art performance while satisfying parameter efficiency. Code and models are available at https://github.com/jiawen-zhu/ViPT.

        ----

        ## [908] Progressive Neighbor Consistency Mining for Correspondence Pruning

        **Authors**: *Xin Liu, Jufeng Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00919](https://doi.org/10.1109/CVPR52729.2023.00919)

        **Abstract**:

        The goal of correspondence pruning is to recognize correct correspondences (inliers) from initial ones, with applications to various feature matching based tasks. Seeking neighbors in the coordinate and feature spaces is a common strategy in many previous methods. However, it is difficult to ensure that these neighbors are always consistent, since the distribution of false correspondences is extremely irregular. For addressing this problem, we propose a novel global-graph space to search for consistent neighbors based on a weighted global graph that can explicitly explore long-range dependencies among correspondences. On top of that, we progressively construct three neighbor embeddings according to different neighbor search spaces, and design a Neighbor Consistency block to extract neighbor context and explore their interactions sequentially. In the end, we develop a Neighbor Consistency Mining Network (NCMNet) for accurately recovering camera poses and identifying inliers. Experimental results indicate that our NCMNet achieves a significant performance advantage over state-of-the-art competitors on challenging outdoor and indoor matching scenes. The source code can be found at https://github.com/xinliu29/NCMNet.

        ----

        ## [909] Geometric Visual Similarity Learning in 3D Medical Image Self-Supervised Pre-training

        **Authors**: *Yuting He, Guanyu Yang, Rongjun Ge, Yang Chen, Jean-Louis Coatrieux, Boyu Wang, Shuo Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00920](https://doi.org/10.1109/CVPR52729.2023.00920)

        **Abstract**:

        Learning inter-image similarity is crucial for 3D medical images self-supervised pre-training, due to their sharing of numerous same semantic regions. However, the lack of the semantic prior in metrics and the semantic-independent variation in 3D medical images make it challenging to get a reliable measurement for the inter-image similarity, hindering the learning of consistent representation for same semantics. We investigate the challenging problem of this task, i.e., learning a consistent representation between images for a clustering effect of same semantic features. We propose a novel visual similarity learning paradigm, Geometric Visual Similarity Learning, which embeds the prior of topological invariance into the measurement of the inter-image similarity for consistent representation of semantic regions. To drive this paradigm, we further construct a novel geometric matching head, the Z-matching head, to collaboratively learn the global and local similarity of semantic regions, guiding the efficient representation learning for different scale-level inter-image semantic features. Our experiments demonstrate that the pre-training with our learning of inter-image similarity yields more powerful inner-scene, inter-scene, and global-local transferring ability on four challenging 3D medical image tasks. Our codes and pre-trained models will be publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/YutingHe-list/GVSL.

        ----

        ## [910] Unsupervised Visible-Infrared Person Re-Identification via Progressive Graph Matching and Alternate Learning

        **Authors**: *Zesen Wu, Mang Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00921](https://doi.org/10.1109/CVPR52729.2023.00921)

        **Abstract**:

        Unsupervised visible-infrared person re-identification is a challenging task due to the large modality gap and the unavailability of cross-modality correspondences. Cross-modality correspondences are very crucial to bridge the modality gap. Some existing works try to mine cross-modality correspondences, but they focus only on local information. They do not fully exploit the global relationship across identities, thus limiting the quality of the mined correspondences. Worse still, the number of clusters of the two modalities is often inconsistent, exacerbating the unreliability of the generated correspondences. In response, we devise a Progressive Graph Matching method to globally mine cross-modality correspondences under cluster imbalance scenarios. PGM formulates correspondence mining as a graph matching process and considers the global information by minimizing the global matching cost, where the matching cost measures the dissimilarity of clusters. Besides, PGM adopts a progressive strategy to address the imbalance issue with multiple dynamic matching processes. Based on PGM, we design an Alternate Cross Contrastive Learning (ACCL) module to reduce the modality gap with the mined cross-modality correspondences, while mitigating the effect of noise in correspondences through an alternate scheme. Extensive experiments demonstrate the reliability of the generated correspondences and the effectiveness of our method.

        ----

        ## [911] Domain Generalized Stereo Matching via Hierarchical Visual Transformation

        **Authors**: *Tianyu Chang, Xun Yang, Tianzhu Zhang, Meng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00922](https://doi.org/10.1109/CVPR52729.2023.00922)

        **Abstract**:

        Recently, deep Stereo Matching (SM) networks have shown impressive performance and attracted increasing attention in computer vision. However, existing deep SM networks are prone to learn dataset-dependent shortcuts, which fail to generalize well on unseen realistic datasets. This paper takes a step towards training robust models for the domain generalized SM task, which mainly focuses on learning shortcut-invariant representation from synthetic data to alleviate the domain shifts. Specifically, we propose a Hierarchical Visual Transformation (HVT) network to 1) first transform the training sample hierarchically into new domains with diverse distributions from three levels: Global, Local, and Pixel, 2) then maximize the visual discrepancy between the source domain and new domains, and minimize the cross-domain feature inconsistency to capture domain-invariant features. In this way, we can prevent the model from exploiting the artifacts of synthetic stereo images as shortcut features, thereby estimating the disparity maps more effectively based on the learned robust and shortcut-invariant representation. We integrate our proposed HVT network with SOTA SM networks and evaluate its effectiveness on several public SM benchmark datasets. Extensive experiments clearly show that the HVT network can substantially enhance the performance of existing SM networks in synthetic-to-realistic domain generalization.

        ----

        ## [912] Unsupervised Cumulative Domain Adaptation for Foggy Scene Optical Flow

        **Authors**: *Hanyu Zhou, Yi Chang, Wending Yan, Luxin Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00923](https://doi.org/10.1109/CVPR52729.2023.00923)

        **Abstract**:

        Optical flow has achieved great success under clean scenes, but suffers from restricted performance under foggy scenes. To bridge the clean-to-foggy domain gap, the existing methods typically adopt the domain adaptation to transfer the motion knowledge from clean to synthetic foggy domain. However, these methods unexpectedly neglect the synthetic-to-real domain gap, and thus are erroneous when applied to real-world scenes. To handle the practical optical flow under real foggy scenes, in this work, we propose a novel unsupervised cumulative domain adaptation optical flow (UCDA-Flow) framework: depth-association motion adaptation and correlation-alignment motion adaptation. Specifically, we discover that depth is a key ingredient to in-fluence the optical flow: the deeper depth, the inferior optical flow, which motivates us to design a depth-association motion adaptation module to bridge the clean-to-foggy domain gap. Moreover, we figure out that the cost volume correlation shares similar distribution of the synthetic and real foggy images, which enlightens us to devise a correlation-alignment motion adaptation module to distill motion knowledge of the synthetic foggy domain to the real foggy domain. Note that synthetic fog is designed as the intermediate domain. Under this unified framework, the proposed cumulative adaptation progressively transfers knowledge from clean scenes to real foggy scenes. Extensive experiments have been performed to verify the superiority of the proposed method.

        ----

        ## [913] PVO: Panoptic Visual Odometry

        **Authors**: *Weicai Ye, Xinyue Lan, Shuo Chen, Yuhang Ming, Xingyuan Yu, Hujun Bao, Zhaopeng Cui, Guofeng Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00924](https://doi.org/10.1109/CVPR52729.2023.00924)

        **Abstract**:

        We present PVO, a novel panoptic visual odometry framework to achieve more comprehensive modeling of the scene motion, geometry, and panoptic segmentation information. Our PVO models visual odometry (VO) and video panoptic segmentation (VPS) in a unified view, which makes the two tasks mutually beneficial. Specifically, we introduce a panoptic update module into the VO Module with the guidance of image panoptic segmentation. This Panoptic-Enhanced VO Module can alleviate the impact of dynamic objects in the camera pose estimation with a panoptic-aware dynamic mask. On the other hand, the VO-Enhanced VPS Module also improves the segmentation accuracy by fusing the panoptic segmentation result of the current frame on the fly to the adjacent frames, using geometric information such as camera pose, depth, and optical flow obtained from the VO Module. These two modules contribute to each other through recurrent iterative optimization. Extensive experiments demonstrate that PVO outperforms state-of-the-art methods in both visual odometry and video panoptic segmentation tasks.

        ----

        ## [914] BAEFormer: Bi-Directional and Early Interaction Transformers for Bird's Eye View Semantic Segmentation

        **Authors**: *Cong Pan, Yonghao He, Junran Peng, Qian Zhang, Wei Sui, Zhaoxiang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00925](https://doi.org/10.1109/CVPR52729.2023.00925)

        **Abstract**:

        Bird's Eye View (BEV) semantic segmentation is a critical task in autonomous driving. However, existing Transformer-based methods confront difficulties in transforming Perspective View (PV) to BEV due to their unidirectional and posterior interaction mechanisms. To address this issue, we propose a novel Bi-directional and Early Interaction Transformers framework named BAEFormer, consisting of (i) an early-interaction PV-BEV pipeline and (ii) a bi-directional cross-attention mechanism. Moreover, we find that the image feature maps' resolution in the cross-attention module has a limited effect on the final performance. Under this critical observation, we propose to enlarge the size of input images and downsample the multi-view image features for cross-interaction, further improving the accuracy while keeping the amount of computation controllable. Our proposed method for BEV semantic segmentation achieves state-of-the-art performance in real-time inference speed on the nuScenes dataset, i.e., 38.9 mIoU at 45 FPS on a single A100 GPU.

        ----

        ## [915] Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark

        **Authors**: *Xiaofeng Wang, Zheng Zhu, Yunpeng Zhang, Guan Huang, Yun Ye, Wenbo Xu, Ziwei Chen, Xingang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00926](https://doi.org/10.1109/CVPR52729.2023.00926)

        **Abstract**:

        In recent years, vision-centric perception has flourished in various autonomous driving tasks, including 3D detection, semantic map construction, motion forecasting, and depth estimation. Nevertheless, the latency of vision-centric approaches is too high for practical deployment (e.g., most camera-based 3D detectors have a runtime greater than 300ms). To bridge the gap between ideal researches and real-world applications, it is necessary to quantify the trade-off between performance and efficiency. Traditionally, autonomous-driving perception benchmarks perform the offline evaluation, neglecting the inference time delay. To mitigate the problem, we propose the Autonomous-driving StreAming Perception (ASAP) benchmark, which is the first benchmark to evaluate the online performance of vision-centric perception in autonomous driving. On the basis of the 2Hz annotated nuScenes dataset, we first propose an annotation-extending pipeline to generate high-frame-rate labels for the 12Hz raw images. Referring to the practical deployment, the Streaming Perception Under constRained-computation (SPUR) evaluation protocol is further constructed, where the 12Hz inputs are utilized for streaming evaluation under the constraints of different computational resources. In the ASAP benchmark, comprehensive experiment results reveal that the model rank alters under different constraints, suggesting that the model latency and computation budget should be considered as design choices to optimize the practical deployment. To facilitate further research, we establish baselines for camera-based streaming 3D detection, which consistently enhance the streaming performance across various hardware. ASAP project page: https://github.com/JeffWang987/ASAP.

        ----

        ## [916] Visual Exemplar Driven Task-Prompting for Unified Perception in Autonomous Driving

        **Authors**: *Xiwen Liang, Minzhe Niu, Jianhua Han, Hang Xu, Chunjing Xu, Xiaodan Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00927](https://doi.org/10.1109/CVPR52729.2023.00927)

        **Abstract**:

        Multi-task learning has emerged as a powerful paradigm to solve a range of tasks simultaneously with good efficiency in both computation resources and inference time. However, these algorithms are designed for different tasks mostly not within the scope of autonomous driving, thus making it hard to compare multi-task methods in autonomous driving. Aiming to enable the comprehensive evaluation of present multi-task learning methods in autonomous driving, we extensively investigate the performance of popular multi-task methods on the large-scale driving dataset, which covers four common perception tasks, i.e., object detection, semantic segmentation, drivable area segmentation, and lane detection. We provide an in-depth analysis of current multi-task learning methods under different common settings and find out that the existing methods make progress but there is still a large performance gap compared with single-task baselines. To alleviate this dilemma in autonomous driving, we present an effective multi-task framework, VE-Prompt, which introduces visual exemplars via task-specific prompting to guide the model toward learning high-quality task-specific representations. Specifically, we generate visual exemplars based on bounding boxes and color-based markers, which provide accurate visual appearances of target categories and further mitigate the performance gap. Furthermore, we bridge transformer-based encoders and convolutional layers for efficient and accurate unified perception in autonomous driving. Comprehensive experimental results on the diverse self-driving dataset BDD100K show that the VE-Prompt improves the multi-task baseline and further surpasses single-task models.

        ----

        ## [917] MIXSIM: A Hierarchical Framework for Mixed Reality Traffic Simulation

        **Authors**: *Simon Suo, Kelvin Wong, Justin Xu, James Tu, Alexander Cui, Sergio Casas, Raquel Urtasun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00928](https://doi.org/10.1109/CVPR52729.2023.00928)

        **Abstract**:

        The prevailing way to test a self-driving vehicle (SDV) in simulation involves non-reactive open-loop replay of real world scenarios. However, in order to safely deploy SDVs to the real world, we need to evaluate them in closed-loop. Towards this goal, we propose to leverage the wealth of interesting scenarios captured in the real world and make them reactive and controllable to enable closed-loop SDV evaluation in what-if situations. In particular, we present MIXSIM, a hierarchical framework for mixed reality traffic simulation. MIXSIM explicitly models agent goals as routes along the road network and learns a reactive route-conditional policy. By inferring each agent's route from the original scenario, MIXSIM can reactively re-simulate the scenario and enable testing different autonomy systems under the same conditions. Furthermore, by varying each agent's route, we can expand the scope of testing to what-if situations with realistic variations in agent behaviors or even safety critical interactions. Our experiments show that MIXSIM can serve as a realistic, reactive, and controllable digital twin of real world scenarios. For more information, please visit the project website: https://waabi.ai/research/mixsim/

        ----

        ## [918] Uncovering the Missing Pattern: Unified Framework Towards Trajectory Imputation and Prediction

        **Authors**: *Yi Xu, Armin Bazarjani, Hyung-Gun Chi, Chiho Choi, Yun Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00929](https://doi.org/10.1109/CVPR52729.2023.00929)

        **Abstract**:

        Trajectory prediction is a crucial undertaking in understanding entity movement or human behavior from observed sequences. However, current methods often assume that the observed sequences are complete while ignoring the potential for missing values caused by object occlusion, scope limitation, sensor failure, etc. This limitation inevitably hinders the accuracy of trajectory prediction. To address this issue, our paper presents a unified framework, the Graph-based Conditional Variational Recurrent Neural Network (GC-VRNN), which can perform trajectory imputation and prediction simultaneously. Specifically, we introduce a novel Multi-Space Graph Neural Network (MS-GNN) that can extract spatial features from incomplete observations and leverage missing patterns. Additionally, we employ a Conditional VRNN with a specifically designed Temporal Decay (TD) module to capture temporal dependencies and temporal missing patterns in incomplete trajectories. The inclusion of the TD module allows for valuable information to be conveyed through the temporal flow. We also curate and benchmark three practical datasets for the joint problem of trajectory imputation and prediction. Extensive experiments verify the exceptional performance of our proposed method. As far as we know, this is the first work to address the lack of benchmarks and techniques for trajectory imputation and prediction in a unified manner.

        ----

        ## [919] MotionDiffuser: Controllable Multi-Agent Motion Prediction Using Diffusion

        **Authors**: *Chiyu Max Jiang, Andre Cornman, Cheolho Park, Benjamin Sapp, Yin Zhou, Dragomir Anguelov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00930](https://doi.org/10.1109/CVPR52729.2023.00930)

        **Abstract**:

        We present MotionDiffuser, a diffusion based representation for the joint distribution of future trajectories over multiple agents. Such representation has several key advantages: first, our model learns a highly multimodal distribution that captures diverse future outcomes. Second, the simple predictor design requires only a single L2 loss training objective, and does not depend on trajectory anchors. Third, our model is capable of learning the joint distribution for the motion of multiple agents in a permutation-invariant manner. Furthermore, we utilize a compressed trajectory representation via PCA, which improves model performance and allows for efficient computation of the exact sample log probability. Subsequently, we propose a general constrained sampling framework that enables controlled trajectory sampling based on differentiable cost functions. This strategy enables a host of applications such as enforcing rules and physical priors, or creating tailored simulation scenarios. MotionDiffuser can be combined with existing backbone architectures to achieve top motion forecasting results. We obtain state-of-the-art results for multi-agent motion prediction on the Waymo Open Motion Dataset.

        ----

        ## [920] Learning Human-to-Robot Handovers from Point Clouds

        **Authors**: *Sammy Joe Christen, Wei Yang, Claudia Pérez-D'Arpino, Otmar Hilliges, Dieter Fox, Yu-Wei Chao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00931](https://doi.org/10.1109/CVPR52729.2023.00931)

        **Abstract**:

        We propose the first framework to learn control policies for vision-based human-to-robot handovers, a critical task for human-robot interaction. While research in Embodied AI has made significant progress in training robot agents in simulated environments, interacting with humans remains challenging due to the difficulties of simulating humans. Fortunately, recent research has developed realistic simulated environments for human-to-robot handovers. Leveraging this result, we introduce a method that is trained with a human-in-the-loop via a two-stage teacher-student framework that uses motion and grasp planning, reinforcement learning, and self-supervision. We show significant performance gains over baselines on a simulation benchmark, sim-to-sim transfer and sim-to-real transfer. Video and code are available at https://handover-sim2real.github.io.

        ----

        ## [921] Phone2Proc: Bringing Robust Robots into Our Chaotic World

        **Authors**: *Matt Deitke, Rose Hendrix, Ali Farhadi, Kiana Ehsani, Aniruddha Kembhavi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00932](https://doi.org/10.1109/CVPR52729.2023.00932)

        **Abstract**:

        Training embodied agents in simulation has become mainstream for the embodied AI community. However, these agents often struggle when deployed in the physical world due to their inability to generalize to real-world environments. In this paper, we present Phone2Proc, a method that uses a 10-minute phone scan and conditional procedural generation to create a distribution of training scenes that are semantically similar to the target environment. The generated scenes are conditioned on the wall layout and arrangement of large objects from the scan, while also sampling lighting, clutter, surface textures, and instances of smaller objects with randomized placement and materials. Leveraging just a simple RGB camera, training with Phone2Proc shows massive improvements from 34.7% to 70.7% success rate in sim-to-real ObjectNav performance across a test suite of over 200 trials in diverse real-world environments, including homes, offices, and RoboTHOR. Furthermore, Phone2Proc's diverse distribution of generated scenes makes agents remarkably robust to changes in the real world, such as human movement, object rearrangement, lighting changes, or clutter.

        ----

        ## [922] GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields

        **Authors**: *Alessandro Ruzzi, Xiangwei Shi, Xi Wang, Gengyan Li, Shalini De Mello, Hyung Jin Chang, Xucong Zhang, Otmar Hilliges*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00933](https://doi.org/10.1109/CVPR52729.2023.00933)

        **Abstract**:

        We propose GazeNeRF, a 3D-aware method for the task of gaze redirection. Existing gaze redirection methods operate on 2D images and struggle to generate 3D consistent results. Instead, we build on the intuition that the face region and eyeballs are separate 3D structures that move in a coordinated yet independent fashion. Our method leverages recent advancements in conditional image-based neural radiance fields and proposes a two-stream architecture that predicts volumetric features for the face and eye regions separately. Rigidly transforming the eye features via a 3D rotation matrix provides fine-grained control over the desired gaze angle. The final, redirected image is then attained via differentiable volume compositing. Our experiments show that this architecture outperforms naively conditioned NeRF baselines as well as previous state-of-the-art 2D gaze redirection methods in terms of redirection accuracy and identity preservation. Code and models will be released for research purposes.

        ----

        ## [923] Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking

        **Authors**: *Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, Kris Kitani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00934](https://doi.org/10.1109/CVPR52729.2023.00934)

        **Abstract**:

        Kalman filter (KF) based methods for multi-object tracking (MOT) make an assumption that objects move linearly. While this assumption is acceptable for very short periods of occlusion, linear estimates of motion for prolonged time can be highly inaccurate. Moreover, when there is no measurement available to update Kalman filter parameters, the standard convention is to trust the priori state estimations for posteriori update. This leads to the accumulation of errors during a period of occlusion. The error causes significant motion direction variance in practice. In this work, we show that a basic Kalman filter can still obtain state-of-the-art tracking performance if proper care is taken to fix the noise accumulated during occlusion. Instead of relying only on the linear state estimate (i.e., estimation-centric approach), we use object observations (i.e., the measurements by object detector) to compute a virtual trajectory over the occlusion period to fix the error accumulation of filter parameters. This allows more time steps to correct errors accumulated during occlusion. We name our method Observation-Centric SORT (OC-SORT). It remains Simple, Online, and Real-Time but improves robustness during occlusion and non-linear motion. Given off-the-shelf detections as input, OC-SORT runs at 700+ FPS on a single CPU. It achieves state-of-the-art on multiple datasets, including MOT17, MOT20, KITTI, head tracking, and especially DanceTrack where the object motion is highly non-linear. The code and models are available at https://github.com/noahcao/OC_SORT.

        ----

        ## [924] Autoregressive Visual Tracking

        **Authors**: *Xing Wei, Yifan Bai, Yongchao Zheng, Dahu Shi, Yihong Gong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00935](https://doi.org/10.1109/CVPR52729.2023.00935)

        **Abstract**:

        We present ARTrack, an autoregressive framework for visual object tracking. ARTrack tackles tracking as a coordinate sequence interpretation task that estimates object trajectories progressively, where the current estimate is induced by previous states and in turn affects subsequences. This time-autoregressive approach models the sequential evolution of trajectories to keep tracing the object across frames, making it superior to existing template matching based trackers that only consider the per-frame localization accuracy. ARTrack is simple and direct, eliminating customized localization heads and post-processings. Despite its simplicity, ARTrack achieves state-of-the-art performance on prevailing benchmark datasets. Source code is available at https://github.com/MIV-XJTU/ARTrack.

        ----

        ## [925] OpenGait: Revisiting Gait Recognition Toward Better Practicality

        **Authors**: *Chao Fan, Junhao Liang, Chuanfu Shen, Saihui Hou, Yongzhen Huang, Shiqi Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00936](https://doi.org/10.1109/CVPR52729.2023.00936)

        **Abstract**:

        Gait recognition is one of the most critical long-distance identification technologies and increasingly gains popularity in both research and industry communities. Despite the significant progress made in indoor datasets, much evidence shows that gait recognition techniques perform poorly in the wild. More importantly, we also find that some conclusions drawn from indoor datasets cannot be generalized to real applications. Therefore, the primary goal of this paper is to present a comprehensive benchmark study for better practicality rather than only a particular model for better performance. To this end, we first develop a flexible and efficient gait recognition codebase named OpenGait. Based on OpenGait, we deeply revisit the recent development of gait recognition by re-conducting the ablative experiments. Encouragingly, we detect some unperfect parts of certain prior woks, as well as new insights. Inspired by these discoveries, we develop a structurally simple, empirically powerful, and practically robust baseline model, Gait-Base. Experimentally, we comprehensively compare Gait-Base with many current gait recognition methods on multiple public datasets, and the results reflect that GaitBase achieves significantly strong performance in most cases regardless of indoor or outdoor situations. Code is available at https://github.com/ShiqiYu/OpenGait.

        ----

        ## [926] Pose-disentangled Contrastive Learning for Self-supervised Facial Representation

        **Authors**: *Yuanyuan Liu, Wenbin Wang, Yibing Zhan, Shaoze Feng, Kejun Liu, Zhe Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00937](https://doi.org/10.1109/CVPR52729.2023.00937)

        **Abstract**:

        Self-supervised facial representation has recently attracted increasing attention due to its ability to perform face understanding without relying on large-scale annotated datasets heavily. However, analytically, current contrastive-based self-supervised learning (SSL) still performs unsatisfactorily for learning facial representation. More specifically, existing contrastive learning (CL) tends to learn pose-invariant features that cannot depict the pose details of faces, compromising the learning performance. To conquer the above limitation of CL, we propose a novel Pose-disentangled Contrastive Learning (PCL) method for general self-supervised facial representation. Our PCL first devises a pose-disentangled decoder (PDD) with a delicately designed orthogonalizing regulation, which disentangles the pose-related features from the face-aware features; therefore, pose-related and other pose-unrelated facial information could be performed in individual subnetworks and do not affect each other's training. Furthermore, we introduce a pose-related contrastive learning scheme that learns pose-related information based on data augmentation of the same image, which would deliver more effective face-aware representation for various downstream tasks. We conducted linear evaluation on four challenging downstream facial understanding tasks, i.e., facial expression recognition, face recognition, AU detection and head pose estimation. Experimental results demonstrate that PCL significantly outperforms cuttingedge SSL methods. Our Code is available at https://github.com/DreamMr/PCL.

        ----

        ## [927] Identity-Preserving Talking Face Generation with Landmark and Appearance Priors

        **Authors**: *Weizhi Zhong, Chaowei Fang, Yinqi Cai, Pengxu Wei, Gangming Zhao, Liang Lin, Guanbin Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00938](https://doi.org/10.1109/CVPR52729.2023.00938)

        **Abstract**:

        Generating talking face videos from audio attracts lots of research interest. A few person-specific methods can generate vivid videos but require the target speaker's videos for training or fine-tuning. Existing person-generic methods have difficulty in generating realistic and lip-synced videos while preserving identity information. To tackle this problem, we propose a two-stage framework consisting of audio-to-landmark generation and landmark-to-video rendering procedures. First, we devise a novel Transformer-based landmark generator to infer lip and jaw landmarks from the audio. Prior landmark characteristics of the speaker's face are employed to make the generated landmarks coincide with the facial outline of the speaker. Then, a video rendering model is built to translate the generated landmarks into face images. During this stage, prior appearance information is extracted from the lower-half occluded target face and static reference images, which helps generate realistic and identity-preserving visual content. For effectively exploring the prior information of static reference images, we align static reference images with the target face's pose and expression based on motion fields. Moreover, auditory features are reused to guarantee that the generated face images are well synchronized with the audio. Extensive experiments demonstrate that our method can produce more realistic, lip-synced, and identity-preserving videos than existing person-generic talking face generation methods.

        ----

        ## [928] DF-Platter: Multi-Face Heterogeneous Deepfake Dataset

        **Authors**: *Kartik Narayan, Harsh Agarwal, Kartik Thakral, Surbhi Mittal, Mayank Vatsa, Richa Singh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00939](https://doi.org/10.1109/CVPR52729.2023.00939)

        **Abstract**:

        Deepfake detection is gaining significant importance in the research community. While most of the research efforts are focused towards high-quality images and videos with controlled appearance of individuals, deepfake generation algorithms now have the capability to generate deep-fakes with low-resolution, occlusion, and manipulation of multiple subjects. In this research, we emulate the real-world scenario of deepfake generation and propose the DF-Platter dataset, which contains (i) both low-resolution and high-resolution deepfakes generated using multiple generation techniques and (ii) single-subject and multiple-subject deepfakes, with face images of Indian ethnicity. Faces in the dataset are annotated for various attributes such as gender, age, skin tone, and occlusion. The dataset is prepared in 116 days with continuous usage of 32 GPUs accounting to 1,800 GB cumulative memory. With over 500 GBs in size, the dataset contains a total of 133,260 videos encompassing three sets. To the best of our knowledge, this is one of the largest datasets containing vast variability and multiple challenges. We also provide benchmark results under multiple evaluation settings using popular and state-of-the-art deepfake detection models, for c0 images and videos along with c23 and c40 compression variants. The results demonstrate a significant performance reduction in the deepfake detection task on low-resolution deep-fakes. Furthermore, existing techniques yield declined detection accuracy on multiple-subject deepfakes. It is our assertion that this database will improve the state-of-the-art by extending the capabilities of deepfake detection algorithms to real-world scenarios. The database is available at: http://iab-rubric.org/df-platter-database.

        ----

        ## [929] Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos

        **Authors**: *Kun Su, Kaizhi Qian, Eli Shlizerman, Antonio Torralba, Chuang Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00940](https://doi.org/10.1109/CVPR52729.2023.00940)

        **Abstract**:

        Modeling sounds emitted from physical object interactions is critical for immersive perceptual experiences in real and virtual worlds. Traditional methods of impact sound synthesis use physics simulation to obtain a set of physics parameters that could represent and synthesize the sound. However, they require fine details of both the object geometries and impact locations, which are rarely available in the real world and can not be applied to synthesize impact sounds from common videos. On the other hand, existing video-driven deep learning-based approaches could only capture the weak correspondence between visual content and impact sounds since they lack of physics knowledge. In this work, we propose a physics-driven diffusion model that can synthesize high-fidelity impact sound for a silent video clip. In addition to the video content, we propose to use additional physics priors to guide the impact sound synthesis procedure. The physics priors include both physics parameters that are directly estimated from noisy real-world impact sound examples without sophisticated setup and learned residual parameters that interpret the sound environment via neural networks. We further implement a novel diffusion model with specific training and inference strategies to combine physics priors and visual information for impact sound synthesis. Experimental results show that our model outperforms several existing systems in generating realistic impact sounds. Lastly, the physics-based representations are fully interpretable and transparent, thus allowing us to perform sound editing flexibly. We encourage the readers visit our project page
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://sukun1045.github.io/video-physics-sound-diffusion/ to watch demo videos with the audio turned on to experience the result.

        ----

        ## [930] MoFusion: A Framework for Denoising-Diffusion-Based Motion Synthesis

        **Authors**: *Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, Christian Theobalt*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00941](https://doi.org/10.1109/CVPR52729.2023.00941)

        **Abstract**:

        Conventional methods for human motion synthesis have either been deterministic or have had to struggle with the trade-off between motion diversity vs motion quality. In response to these limitations, we introduce MoFusion, i.e., a new denoising-diffusion-based framework for high-quality conditional human motion synthesis that can synthesise long, temporally plausible, and semantically accurate motions based on a range of conditioning contexts (such as music and text). We also present ways to introduce well-known kinematic losses for motion plausibility within the motion-diffusion framework through our scheduled weighting strategy. The learned latent space can be used for several interactive motion-editing applications like in-betweening, seed-conditioning, and text-based editing, thus, providing crucial abilities for virtual-character animation and robotics. Through comprehensive quantitative evaluations and a perceptual user study, we demonstrate the effectiveness of MoFusion compared to the state of the art on established benchmarks in the literature. We urge the reader to watch our supplementary video at https://vcai.mpi-inf.mpg.de/projects/MoFusion/.

        ----

        ## [931] Adaptive Global Decay Process for Event Cameras

        **Authors**: *Urbano Miguel Nunes, Ryad Benosman, Sio-Hoi Ieng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00942](https://doi.org/10.1109/CVPR52729.2023.00942)

        **Abstract**:

        In virtually all event-based vision problems, there is the need to select the most recent events, which are assumed to carry the most relevant information content. To achieve this, at least one of three main strategies is applied, namely: 1) constant temporal decay or fixed time window, 2) constant number of events, and 3) flow-based lifetime of events. However, these strategies suffer from at least one major limitation each. We instead propose a novel decay process for event cameras that adapts to the global scene dynamics and whose latency is in the order of nanoseconds. The main idea is to construct an adaptive quantity that encodes the global scene dynamics, denoted by event activity. The proposed method is evaluated in several event-based vision problems and datasets, consistently improving the corresponding baseline methods' performance. We thus believe it can have a significant widespread impact on event-based research. Code available: https://github.com/neuromorphic-paris/event.batch.

        ----

        ## [932] Frame-Event Alignment and Fusion Network for High Frame Rate Tracking

        **Authors**: *Jiqing Zhang, Yuanchen Wang, Wenxi Liu, Meng Li, Jinpeng Bai, Baocai Yin, Xin Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00943](https://doi.org/10.1109/CVPR52729.2023.00943)

        **Abstract**:

        Most existing RGB-based trackers target low frame rate benchmarks of around 30 frames per second. This setting restricts the tracker's functionality in the real world, especially for fast motion. Event-based cameras as bioinspired sensors provide considerable potential for high frame rate tracking due to their high temporal resolution. However, event-based cameras cannot offer fine-grained texture information like conventional cameras. This unique complementarity motivates us to combine conventional frames and events for high frame rate object tracking under various challenging conditions. In this paper, we propose an end-to-end network consisting of multi-modality alignment and fusion modules to effectively combine meaningful information from both modalities at different measurement rates. The alignment module is responsible for cross-style and cross-frame-rate alignment between frame and event modalities under the guidance of the moving cues furnished by events. While the fusion module is accountable for emphasizing valuable features and suppressing noise information by the mutual complement between the two modalities. Extensive experiments show that the proposed approach outper-forms state-of-the-art trackers by a significant margin in high frame rate tracking. With the FE240Hz dataset, our approach achieves high frame rate tracking up to 240Hz.

        ----

        ## [933] Exploring Discontinuity for Video Frame Interpolation

        **Authors**: *Sangjin Lee, Hyeongmin Lee, Chajin Shin, Hanbin Son, Sangyoun Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00944](https://doi.org/10.1109/CVPR52729.2023.00944)

        **Abstract**:

        Video frame interpolation (VFI) is the task that synthesizes the intermediate frame given two consecutive frames. Most of the previous studies have focused on appropriate frame warping operations and refinement modules for the warped frames. These studies have been conducted on natural videos containing only continuous motions. However, many practical videos contain various unnatural objects with discontinuous motions such as logos, user interfaces and subtitles. We propose three techniques that can make the existing deep learning-based VFI architectures robust to these elements. First is a novel data augmentation strategy called figure-text mixing (FTM) which can make the models learn discontinuous motions during training stage without any extra dataset. Second, we propose a simple but effective module that predicts a map called discontinuity map (D-map), which densely distinguishes between areas of continuous and discontinuous motions. Lastly, we propose loss functions to give supervisions of the discontinuous motion areas which can be applied along with FTM and D-map. We additionally collect a special test benchmark called Graphical Discontinuous Motion (GDM) dataset consisting of some mobile games and chatting videos. Applied to the various state-of-the-art VFI networks, our method significantly improves the interpolation qualities on the videos from not only GDM dataset, but also the existing benchmarks containing only continuous motions such as Vimeo90K, UCF101, and DAVIS.

        ----

        ## [934] AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation

        **Authors**: *Zhen Li, Zuo-Liang Zhu, Linghao Han, Qibin Hou, Chun-Le Guo, Ming-Ming Cheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00945](https://doi.org/10.1109/CVPR52729.2023.00945)

        **Abstract**:

        We present All-Pairs Multi-Field Transforms (AMT), a new network architecture for video frame interpolation. It is based on two essential designs. First, we build bidirectional correlation volumes for all pairs of pixels, and use the predicted bilateral flows to retrieve correlations for updating both flows and the interpolated content feature. Second, we derive multiple groups of fine-grained flow fields from one pair of updated coarse flows for performing backward warping on the input frames separately. Combining these two designs enables us to generate promising task-oriented flows and reduce the difficulties in modeling large motions and handling occluded areas during frame interpolation. These qualities promote our model to achieve state-of-the-art performance on various benchmarks with high efficiency. Moreover, our convolution-based model competes favorably compared to Transformer-based models in terms of accuracy and efficiency. Our code is available at https://github.com/MCG-NKU/AMT.

        ----

        ## [935] Frame Interpolation Transformer and Uncertainty Guidance

        **Authors**: *Markus Plack, Matthias B. Hullin, Karlis Martins Briedis, Markus H. Gross, Abdelaziz Djelouah, Christopher Schroers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00946](https://doi.org/10.1109/CVPR52729.2023.00946)

        **Abstract**:

        Video frame interpolation has seen important progress in recent years, thanks to developments in several directions. Some works leverage better optical flow methods with improved splatting strategies or additional cues from depth, while others have investigated alternative approaches through direct predictions or transformers. Still, the problem remains unsolved in more challenging conditions such as complex lighting or large motion. In this work, we are bridging the gap towards video production with a novel transformer-based interpolation network architecture capable of estimating the expected error together with the interpolated frame. This offers several advantages that are of key importance for frame interpolation usage: First, we obtained improved visual quality over several datasets. The improvement in terms of quality is also clearly demonstrated through a user study. Second, our method estimates error maps for the interpolated frame, which are essential for real-life applications on longer video sequences where problematic frames need to be flagged. Finally, for rendered content a partial rendering pass of the intermediate frame, guided by the predicted error, can be utilized during the interpolation to generate a new frame of superior quality. Through this error estimation, our method can produce even higher-quality intermediate frames using only a fraction of the time compared to a full rendering.

        ----

        ## [936] A Simple Baseline for Video Restoration with Grouped Spatial-Temporal Shift

        **Authors**: *Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00947](https://doi.org/10.1109/CVPR52729.2023.00947)

        **Abstract**:

        Video restoration, which aims to restore clear frames from degraded videos, has numerous important applications. The key to video restoration depends on utilizing inter-frame information. However, existing deep learning methods often rely on complicated network architectures, such as optical flow estimation, deformable convo-lution, and cross-frame self-attention layers, resulting in high computational costs. In this study, we propose a sim-ple yet effective framework for video restoration. Our approach is based on grouped spatial-temporal shift, which is a lightweight and straightforward technique that can implicitly capture inter-frame correspondences for multi-frame aggregation. By introducing grouped spatial shift, we attain expansive effective receptive fields. Combined with basic 2D convolution, this simple framework can effectively aggregate inter-frame information. Extensive experiments demonstrate that our framework outperforms the previous state-of-the-art method, while using less than a quarter of its computational cost, on both video deblurring and video denoising tasks. These results indicate the potential for our approach to significantly reduce computational overhead while maintaining high-quality results. Code is avaliable at https://github.com/dasonglil/Shift-Net.

        ----

        ## [937] Recurrent Homography Estimation Using Homography-Guided Image Warping and Focus Transformer

        **Authors**: *Si-Yuan Cao, Runmin Zhang, Lun Luo, Beinan Yu, Zehua Sheng, Junwei Li, Hui-Liang Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00948](https://doi.org/10.1109/CVPR52729.2023.00948)

        **Abstract**:

        We propose the Recurrent homography estimation framework using Homography-guided image Warping and Focus transformer (FocusFormer), named RHWF. Both being appropriately absorbed into the recurrent framework, the homography-guided image warping progressively enhances the feature consistency and the attention-focusing mechanism in FocusFormer aggregates the intra-inter correspondence in a global→nonlocal→local manner. Thanks to the above strategies, RHWF ranks top in accuracy on a variety of datasets, including the challenging cross-resolution and cross-modal ones. Meanwhile, benefiting from the recurrent framework, RHWF achieves parameter efficiency despite the transformer architecture. Compared to previous state-of-the-art approaches LocalTrans and IHN, RHWF reduces the mean average corner error (MACE) by about 70% and 38.1% on the MSCOCO dataset, while saving the parameter costs by 86.5% and 24.6%. Similar to the previous works, RHWF can also be arranged in 1-scale for efficiency and 2-scale for accuracy, with the 1-scale RHWF already outperforming most of the previous methods. Source code is available at https://github.com/imdump178/RHWF.

        ----

        ## [938] HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering

        **Authors**: *Bang-Dang Pham, Phong Tran, Anh Tran, Cuong Pham, Rang Nguyen, Minh Hoai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00949](https://doi.org/10.1109/CVPR52729.2023.00949)

        **Abstract**:

        We consider the challenging task of training models for image-to-video deblurring, which aims to recover a sequence of sharp images corresponding to a given blurry image input. A critical issue disturbing the training of an image-to-video model is the ambiguity of the frame ordering since both the forward and backward sequences are plausible solutions. This paper proposes an effective self-supervised ordering scheme that allows training highquality image-to-video deblurring models. Unlike previous methods that rely on order-invariant losses, we assign an explicit order for each video sequence, thus avoiding the order-ambiguity issue. Specifically, we map each video sequence to a vector in a latent high-dimensional space so that there exists a hyperplane such that for every video sequence, the vectors extracted from it and its reversed sequence are on different sides of the hyperplane. The side of the vectors will be used to define the order of the corresponding sequence. Last but not least, we propose a real-image dataset for the image-to-video deblurring problem that covers a variety of popular domains, including face, hand, and street. Extensive experimental results confirm the effectiveness of our method. Code and data are available at https://github.com/VinAIResearch/HyperCUT.git

        ----

        ## [939] Indescribable Multi-Modal Spatial Evaluator

        **Authors**: *Lingke Kong, X. Sharon Qi, Qijin Shen, Jiacheng Wang, Jingyi Zhang, Yanle Hu, Qichao Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00950](https://doi.org/10.1109/CVPR52729.2023.00950)

        **Abstract**:

        Multi-modal image registration spatially aligns two images with different distributions. One of its major challenges is that images acquired from different imaging machines have different imaging distributions, making it difficult to focus only on the spatial aspect of the images and ignore differences in distributions. In this study, we developed a self-supervised approach, Indescribable Multi-model Spatial Evaluator (IMSE), to address multi-modal image registration. IMSE creates an accurate multi-modal spatial evaluator to measure spatial differences between two images, and then optimizes registration by minimizing the error predicted of the evaluator. To optimize IMSE performance, we also proposed a new style enhancement method called Shuffle Remap which randomizes the image distribution into multiple segments, and then randomly disorders and remaps these segments, so that the distribution of the original image is changed. Shuffle Remap can help IMSE to predict the difference in spatial location from unseen target distributions. Our results show that IMSE outperformed the existing methods for registration using T1-T2 and CT-MRI datasets. IMSE also can be easily integrated into the traditional registration process, and can provide a convenient way to evaluate and visualize registration results. IMSE also has the potential to be used as a new paradigm for image-to-image translation. Our code is available at https://github.com/Kid-Liet/IMSE.

        ----

        ## [940] Structured Kernel Estimation for Photon-Limited Deconvolution

        **Authors**: *Yash Sanghvi, Zhiyuan Mao, Stanley H. Chan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00951](https://doi.org/10.1109/CVPR52729.2023.00951)

        **Abstract**:

        Images taken in a low light condition with the presence of camera shake suffer from motion blur and photon shot noise. While state-of-the-art image restoration networks show promising results, they are largely limited to well-illuminated scenes and their performance drops significantly when photon shot noise is strong. In this paper, we propose a new blur estimation technique customized for photon-limited conditions. The proposed method employs a gradient-based backpropagation method to estimate the blur kernel. By modeling the blur kernel using a low-dimensional representation with the key points on the motion trajectory, we significantly reduce the search space and improve the regularity of the kernel estimation problem. When plugged into an iterative framework, our novel low-dimensional representation provides improved kernel estimates and hence significantly better deconvolution performance when compared to end-to-end trained neural networks. The source code and pretrained mdoels are available at https://github.com/sanghviyashiitb/structured-kernel-cvpr23

        ----

        ## [941] Polarized Color Image Denoising

        **Authors**: *Zhuoxiao Li, Haiyang Jiang, Mingdeng Cao, Yinqiang Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00952](https://doi.org/10.1109/CVPR52729.2023.00952)

        **Abstract**:

        Single-chip polarized color photography provides both visual textures and object surface information in one snapshot. However, the use of an additional directional polarizing filter array tends to lower photon count and SNR, when compared to conventional color imaging. As a result, such a bilayer structure usually leads to unpleasant noisy images and undermines performance of polarization analysis, especially in low-light conditions. It is a challenge for traditional image processing pipelines owing to the fact that the physical constraints exerted implicitly in the channels are excessively complicated. In this paper, we propose to tackle this issue through a noise modeling method for realistic data synthesis and a powerful network structure inspired by vision Transformer. A real-world polarized color image dataset of paired raw short-exposed noisy images and long-exposed reference images is captured for experimental evaluation, which has demonstrated the effectiveness of our approaches for data synthesis and polarized color image denoising. The code and data can be found at https://github.com/bandasyou/pcdenoise.

        ----

        ## [942] Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior

        **Authors**: *Xiaole Tang, Xile Zhao, Jun Liu, Jianli Wang, Yuchun Miao, Tieyong Zeng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00953](https://doi.org/10.1109/CVPR52729.2023.00953)

        **Abstract**:

        Non-blind deblurring methods achieve decent performance under the accurate blur kernel assumption. Since the kernel uncertainty (i.e. kernel error) is inevitable in practice, semi-blind deblurring is suggested to handle it by introducing the prior of the kernel (or induced) error. However, how to design a suitable prior for the kernel (or induced) error remains challenging. Hand-crafted prior, incorporating domain knowledge, generally performs well but may lead to poor performance when kernel (or induced) error is complex. Data-driven prior, which excessively depends on the diversity and abundance of training data, is vulnerable to out-of-distribution blurs and images. To address this challenge, we suggest a dataset-free deep residual prior for the kernel induced error (termed as residual) expressed by a customized untrained deep neural network, which allows us to flexibly adapt to different blurs and images in real scenarios. By organically integrating the respective strengths of deep priors and hand-crafted priors, we propose an unsupervised semi-blind deblurring model which recovers the clear image from the blurry image and inaccurate blur kernel. To tackle the formulated model, an efficient alternating minimization algorithm is developed. Extensive experiments demonstrate the favorable performance of the proposed method as compared to model-driven and data-driven methods in terms of image quality and the robustness to different types of kernel error.

        ----

        ## [943] Low-Light Image Enhancement via Structure Modeling and Guidance

        **Authors**: *Xiaogang Xu, Ruixing Wang, Jiangbo Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00954](https://doi.org/10.1109/CVPR52729.2023.00954)

        **Abstract**:

        This paper proposes a new framework for low-light image enhancement by simultaneously conducting the appearance as well as structure modeling. It employs the structural feature to guide the appearance enhancement, leading to sharp and realistic results. The structure modeling in our framework is implemented as the edge detection in low-light images. It is achieved with a modified generative model via designing a structure-aware feature extractor and generator. The detected edge maps can accurately emphasize the essential structural information, and the edge prediction is robust towards the noises in dark areas. Moreover, to improve the appearance modeling, which is implemented with a simple U-Net, a novel structure-guided enhancement module is proposed with structure-guided feature synthesis layers. The appearance modeling, edge detector, and enhancement module can be trained end-to-end. The experiments are conducted on representative datasets (sRGB and RAW domains), showing that our model consistently achieves SOTA performance on all datasets with the same architecture. The code is available at https://github.com/xiaogangOO/SMG-LLIE.

        ----

        ## [944] Learning Sample Relationship for Exposure Correction

        **Authors**: *Jie Huang, Feng Zhao, Man Zhou, Jie Xiao, Naishan Zheng, Kaiwen Zheng, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00955](https://doi.org/10.1109/CVPR52729.2023.00955)

        **Abstract**:

        Exposure correction task aims to correct the underexposure and its adverse overexposure images to the normal exposure in a single network. As well recognized, the optimization flow is the opposite. Despite great advancement, existing exposure correction methods are usually trained with a mini-batch of both underexposure and overexposure mixed samples and have not explored the relationship between them to solve the optimization inconsistency. In this paper, we introduce a new perspective to conjunct their optimization processes by correlating and constraining the relationship of correction procedure in a mini-batch. The core designs of our framework consist of two steps: 1) formulating the exposure relationship of samples across the batch dimension via a context-irrelevant pretext task. 2) delivering the above sample relationship design as the regularization term within the loss function to promote optimization consistency. The proposed sample relationship design as a general term can be easily integrated into existing exposure correction methods without any computational burden in inference time. Extensive experiments over multiple representative exposure correction benchmarks demonstrate consistent performance gains by introducing our sample relationship design.

        ----

        ## [945] Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising

        **Authors**: *Junyi Li, Zhilu Zhang, Xiaoyu Liu, Chaoyu Feng, Xiaotao Wang, Lei Lei, Wangmeng Zuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00956](https://doi.org/10.1109/CVPR52729.2023.00956)

        **Abstract**:

        Significant progress has been made in self-supervised image denoising (SSID) in the recent few years. However, most methods focus on dealing with spatially independent noise, and they have little practicality on real-world sRGB images with spatially correlated noise. Although pixel-shuffle downsampling has been suggested for breaking the noise correlation, it breaks the original information of images, which limits the denoising performance. In this paper, we propose a novel perspective to solve this problem, i.e., seeking for spatially adaptive supervision for real-world sRGB image denoising. Specifically, we take into account the respective characteristics of flat and textured regions in noisy images, and construct supervisions for them separately. For flat areas, the supervision can be safely derived from non-adjacent pixels, which are much far from the current pixel for excluding the influence of the noise-correlated ones. And we extend the blind-spot network to a blind-neighborhood network (BNN) for providing supervision on flat areas. For textured regions, the supervision has to be closely related to the content of adjacent pixels. And we present a locally aware network (LAN) to meet the requirement, while LAN itself is selectively supervised with the output of BNN. Combining these two supervisions, a denoising network (e.g., U-Net) can be well-trained. Extensive experiments show that our method performs favorably against state-of-the-art SSID methods on real-world sRGB photographs. The code is available at https://github.com/nagejacob/SpatiallyAdaptiveSSID.

        ----

        ## [946] Quantum-Inspired Spectral-Spatial Pyramid Network for Hyperspectral Image Classification

        **Authors**: *Jie Zhang, Yongshan Zhang, Yicong Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00957](https://doi.org/10.1109/CVPR52729.2023.00957)

        **Abstract**:

        Hyperspectral image (HSI) classification aims at assigning a unique label for every pixel to identify categories of different land covers. Existing deep learning models for HSIs are usually performed in a traditional learning paradigm. Being emerging machines, quantum computers are limited in the noisy intermediate-scale quantum (NISQ) era. The quantum theory offers a new paradigm for designing deep learning models. Motivated by the quantum circuit (QC) model, we propose a quantum-inspired spectral-spatial network (QSSN) for HSI feature extraction. The proposed QSSN consists of a phase-prediction module (PPM) and a measurement-like fusion module (MFM) inspired from quantum theory to dynamically fuse spectral and spatial information. Specifically, QSSN uses a quantum representation to represent an HSI cuboid and extracts joint spectral-spatial features using MFM. An HSI cuboid and its phases predicted by PPM are used in the quantum representation. Using QSSN as the building block, we further propose an end-to-end quantum-inspired spectral-spatial pyramid network (QSSPN) for HSI feature extraction and classification. In this pyramid framework, QSSPN progressively learns feature representations by cascading QSSN blocks and performs classification with a softmax classifier. It is the first attempt to introduce quantum theory in HSI processing model design. Substantial experiments are conducted on three HSI datasets to verify the superiority of the proposed QSSPN framework over the state-of-the-art methods.

        ----

        ## [947] Generative Diffusion Prior for Unified Image Restoration and Enhancement

        **Authors**: *Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, Bo Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00958](https://doi.org/10.1109/CVPR52729.2023.00958)

        **Abstract**:

        Existing image restoration methods mostly leverage the posterior distribution of natural images. However, they often assume known degradation and also require supervised training, which restricts their adaptation to complex real applications. In this work, we propose the Generative Diffusion Prior (GDP) to effectively model the posterior distributions in an unsupervised sampling manner. GDP utilizes a pre-train denoising diffusion generative model (DDPM) for solving linear inverse, non-linear, or blind problems. Specifically, GDP systematically explores a protocol of conditional guidance, which is verified more practical than the commonly used guidance way. Furthermore, GDP is strength at optimizing the parameters of degradation model during the denoising process, achieving blind image restoration. Besides, we devise hierarchical guidance and patch-based methods, enabling the GDP to generate images of arbitrary resolutions. Experimentally, we demonstrate GDP's versatility on several image datasets for linear problems, such as super-resolution, deblurring, inpainting, and colorization, as well as non-linear and blind issues, such as low-light enhancement and HDR image recovery. GDP outperforms the current leading unsupervised methods on the diverse benchmarks in reconstruction quality and perceptual quality. Moreover, GDP also generalizes well for natural images or synthesized images with arbitrary sizes from various tasks out of the distribution of the ImageNet training set. The project page is available at https://generativediffusionprior.github.io/

        ----

        ## [948] Ground-Truth Free Meta-Learning for Deep Compressive Sampling

        **Authors**: *Xinran Qin, Yuhui Quan, Tongyao Pang, Hui Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00959](https://doi.org/10.1109/CVPR52729.2023.00959)

        **Abstract**:

        Compressive sampling (CS) is an efficient technique for imaging. This paper proposes a ground-truth (GT) free meta-learning method for CS, which leverages both ex-ternal and internal deep learning for unsupervised high-quality image reconstruction. The proposed method first trains a deep neural network (NN) via external meta-learning using only CS measurements, and then efficiently adapts the trained model to a test sample for exploiting sample-specific internal characteristic for performance gain. The meta-learning and model adaptation are built on an improved Stein's unbiased risk estimator (iSURE) that provides efficient computation and effective guidance for accurate prediction in the range space of the adjoint of the measurement matrix. To improve the learning and adaption on the null space of the measurement matrix, a modi-fied model-agnostic meta-learning scheme and a null-space consistency loss are proposed. In addition, a bias tuning scheme for unrolling NNs is introduced for further acceler-ation of model adaption. Experimental results have demonstrated that the proposed GT-free method performs well and can even compete with supervised methods.

        ----

        ## [949] Recognizability Embedding Enhancement for Very Low-Resolution Face Recognition and Quality Estimation

        **Authors**: *Jacky Chen Long Chai, Tiong-Sik Ng, Cheng-Yaw Low, Jaewoo Park, Andrew Beng Jin Teoh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00960](https://doi.org/10.1109/CVPR52729.2023.00960)

        **Abstract**:

        Very low-resolution face recognition (VLRFR) poses unique challenges, such as tiny regions of interest and poor resolution due to extreme standoff distance or wide viewing angle of the acquisition devices. In this paper, we study principled approaches to elevate the recognizability of a face in the embedding space instead of the visual quality. We first formulate a robust learning-based face recognizability measure, namely recognizability index (RI), based on two criteria: (i) proximity of each face embedding against the unrecognizable faces cluster center and (ii) closeness of each face embedding against its positive and negative class prototypes. We then devise an index diversion loss to push the hard-to-recognize face embedding with low RI away from unrecognizable faces cluster to boost the RI, which reflects better recognizability. Additionally, a perceptibility attention mechanism is introduced to attend to the most recognizable face regions, which offers better explanatory and discriminative traits for embedding learning. Our proposed model is trained end-to-end and simultaneously serves recognizability-aware embedding learning and face quality estimation. To address VLRFR, our extensive evaluations on three challenging low-resolution datasets and face quality assessment demonstrate the superiority of the proposed model over the state-of-the-art methods.

        ----

        ## [950] An Image Quality Assessment Dataset for Portraits

        **Authors**: *Nicolas Chahine, Ana-Stefania Calarasanu, Davide Garcia-Civiero, Théo Cayla, Sira Ferradans, Jean Ponce*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00961](https://doi.org/10.1109/CVPR52729.2023.00961)

        **Abstract**:

        Year after year, the demand for ever-better smartphone photos continues to grow, in particular in the domain of portrait photography. Manufacturers thus use perceptual quality criteria throughout the development of smartphone cameras. This costly procedure can be partially replaced by automated learning-based methods for image quality assessment (IQA). Due to its subjective nature, it is necessary to estimate and guarantee the consistency of the IQA process, a characteristic lacking in the mean opinion scores (MOS) widely used for crowdsourcing IQA. In addition, existing blind IQA (BIQA) datasets pay little attention to the difficulty of cross-content assessment, which may degrade the quality of annotations. This paper introduces PIQ23, a portrait-specific IQA dataset of 5116 images of 50 predefined scenarios acquired by 100 smartphones, covering a high variety of brands, models, and use cases. The dataset includes individuals of various genders and ethnicities who have given explicit and informed consent for their photographs to be used in public research. It is annotated by pairwise comparisons (PWC) collected from over 30 image quality experts for three image attributes: face detail preservation, face target exposure, and overall image quality. An in-depth statistical analysis of these annotations allows us to evaluate their consistency over PIQ23. Finally, we show through an extensive comparison with existing baselines that semantic information (image context) can be used to improve IQA predictions. The dataset along with the proposed statistical analysis and BIQA algorithms are available: https://github.com/DXOMARK-Research/PIQ2023

        ----

        ## [951] Bitstream-Corrupted JPEG Images are Restorable: Two-stage Compensation and Alignment Framework for Image Restoration

        **Authors**: *Wenyang Liu, Yi Wang, Kim-Hui Yap, Lap-Pui Chau*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00962](https://doi.org/10.1109/CVPR52729.2023.00962)

        **Abstract**:

        In this paper, we study a real-world JPEG image restoration problem with bit errors on the encrypted bitstream. The bit errors bring unpredictable color casts and block shifts on decoded image contents, which cannot be resolved by existing image restoration methods mainly relying on pre-defined degradation models in the pixel domain. To address these challenges, we propose a robust JPEG decoder, followed by a two-stage compensation and alignment framework to restore bitstream-corrupted JPEC images. Specifically, the robust JPEC decoder adopts an error-resilient mechanism to decode the corrupted JPEG bitstream. The two-stage framework is composed of the self-compensation and alignment (SCA) stage and the guided-compensation and alignment (GCA) stage. The SCA adaptively performs block-wise image color compensation and alignment based on the estimated color and block offsets via image content similarity. The GCA leverages the extracted low-resolution thumbnail from the JPEG header to guide full-resolution pixel-wise image restoration in a coarse-to-fine manner. It is achieved by a coarse-guided pix2pix network and a refine-guided bi-directional Laplacian pyramid fusion network. We conduct experiments on three benchmarks with varying degrees of bit error rates. Experimental results and ablation studies demonstrate the superiority of our proposed method. The code will be released at https://github.com/wenyang001/Two-ACIR.

        ----

        ## [952] Image Super-Resolution Using T-Tetromino Pixels

        **Authors**: *Simon Grosche, Andy Regensky, Jürgen Seiler, André Kaup*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00963](https://doi.org/10.1109/CVPR52729.2023.00963)

        **Abstract**:

        For modern high-resolution imaging sensors, pixel binning is performed in low-lighting conditions and in case high frame rates are required. To recover the original spatial resolution, single-image super-resolution techniques can be applied for upscaling. To achieve a higher image quality after upscaling, we propose a novel binning concept using tetromino-shaped pixels. It is embedded into the field of compressed sensing and the coherence is calculated to motivate the sensor layouts used. Next, we investigate the reconstruction quality using tetromino pixels for the first time in literature. Instead of using different types of tetrominoes as proposed elsewhere, we show that using a small repeating cell consisting of only four T-tetrominoes is sufficient. For reconstruction, we use a locally fully connected reconstruction (LFCR) network as well as two classical reconstruction methods from the field of compressed sensing. Using the LFCR network in combination with the proposed tetromino layout, we achieve superior image quality in terms of PSNR, SSIM, and visually compared to conventional single-image super-resolution using the very deep super-resolution (VDSR) network. For PSNR, a gain of up to +1.92 dB is achieved.

        ----

        ## [953] CUF: Continuous Upsampling Filters

        **Authors**: *Cristina N. Vasconcelos, A. Cengiz Öztireli, Mark J. Matthews, Milad Hashemi, Kevin Swersky, Andrea Tagliasacchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00964](https://doi.org/10.1109/CVPR52729.2023.00964)

        **Abstract**:

        Neural fields have rapidly been adopted for representing 3D signals, but their application to more classical 2D image-processing has been relatively limited. In this paper, we consider one of the most important operations in image processing: upsampling. In deep learning, learnable upsampling layers have extensively been used for single image super-resolution. We propose to parameterize upsampling kernels as neural fields. This parameterization leads to a compact architecture that obtains a 40-fold reduction in the number of parameters when compared with competing arbitrary-scale super-resolution architectures. When upsampling images of size 256×256 we show that our architecture is 2x-10x more efficient than competing arbitrary-scale super-resolution architectures, and more efficient than sub-pixel convolutions when instantiated to a single-scale model. In the general setting, these gains grow polynomially with the square of the target scale. We validate our method on standard benchmarks showing such efficiency gains can be achieved without sacrifices in super-resolution performance. https://cuf-paper.github.io

        ----

        ## [954] OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution

        **Authors**: *Gaochao Song, Qian Sun, Luo Zhang, Ran Su, Jianfeng Shi, Ying He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00965](https://doi.org/10.1109/CVPR52729.2023.00965)

        **Abstract**:

        Arbitrary-scale image super-resolution (SR) is often tackled using the implicit neural representation (INR) approach, which relies on a position encoding scheme to im-prove its representation ability. In this paper, we introduce orthogonal position encoding (OPE), an extension of po-sition encoding, and an OPE-Upscale module to replace the INR-based upsampling module for arbitrary-scale im-age super-resolution. Our OPE-Upscale module takes 2D coordinates and latent code as inputs, just like INR, but does not require any training parameters. This parameter-free feature allows the OPE-Upscale module to directly perform linear combination operations, resulting in con-tinuous image reconstruction and achieving arbitrary-scale image reconstruction. As a concise SR framework, our method is computationally efficient and consumes less mem-ory than state-of-the-art methods, as confirmed by exten-sive experiments and evaluations. In addition, our method achieves comparable results with state-of-the-art methods in arbitrary-scale image super-resolution. Lastly, we show that OPE corresponds to a set of orthogonal basis, validating our design principle.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://github.com/gaochao-s/ope-sr

        ----

        ## [955] Implicit Diffusion Models for Continuous Super-Resolution

        **Authors**: *Sicheng Gao, Xuhui Liu, Bohan Zeng, Sheng Xu, Yanjing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, Baochang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00966](https://doi.org/10.1109/CVPR52729.2023.00966)

        **Abstract**:

        Image super-resolution (SR) has attracted increasing attention due to its widespread applications. However, current SR methods generally suffer from over-smoothing and artifacts, and most work only with fixed magnifications. This paper introduces an Implicit Diffusion Model (IDM) for high-fidelity continuous image super-resolution. IDM integrates an implicit neural representation and a denoising diffusion model in a unified end-to-end framework, where the implicit neural representation is adopted in the decoding process to learn continuous-resolution representation. Furthermore, we design a scale-adaptive conditioning mechanism that consists of a low-resolution (LR) conditioning network and a scaling factor. The scaling factor regulates the resolution and accordingly modulates the proportion of the LR information and generated features in the final output, which enables the model to accommodate the continuous-resolution requirement. Extensive experiments validate the effectiveness of our IDM and demonstrate its superior performance over prior arts. The source code will be available at https://github.com/Ree1s/IDM.

        ----

        ## [956] Pixels, Regions, and Objects: Multiple Enhancement for Salient Object Detection

        **Authors**: *Yi Wang, Ruili Wang, Xin Fan, Tianzhu Wang, Xiangjian He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00967](https://doi.org/10.1109/CVPR52729.2023.00967)

        **Abstract**:

        Salient object detection (SOD) aims to mimic the human visual system (HVS) and cognition mechanisms to identify and segment salient objects. However, due to the complexity of these mechanisms, current methods are not perfect. Accuracy and robustness need to be further improved, particularly in complex scenes with multiple objects and background clutter. To address this issue, we propose a novel approach called Multiple Enhancement Network (MENet) that adopts the boundary sensibility, content integrity, iterative refinement, and frequency decomposition mechanisms of HVS. A multi-level hybrid loss is firstly designed to guide the network to learn pixel-level, region-level, and object-level features. A flexible multiscale feature enhancement module (ME-Module) is then designed to gradually aggregate and refine global or detailed features by changing the size order of the input feature sequence. An iterative training strategy is used to enhance boundary features and adaptive features in the dual-branch decoder of MENet. Comprehensive evaluations on six challenging benchmark datasets show that MENet achieves state-of-the-art results. Both the codes and results are publicly available at https://github.com/yiwangtz/MENet.

        ----

        ## [957] VILA: Learning Image Aesthetics from User Comments with Vision-Language Pretraining

        **Authors**: *Junjie Ke, Keren Ye, Jiahui Yu, Yonghui Wu, Peyman Milanfar, Feng Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00968](https://doi.org/10.1109/CVPR52729.2023.00968)

        **Abstract**:

        Assessing the aesthetics of an image is challenging, as it is influenced by multiple factors including composition, color, style, and high-level semantics. Existing image aesthetic assessment (IAA) methods primarily rely on human-labeled rating scores, which oversimplify the visual aesthetic information that humans perceive. Conversely, user comments offer more comprehensive information and are a more natural way to express human opinions and preferences regarding image aesthetics. In light of this, we propose learning image aesthetics from user comments, and exploring vision-language pretraining methods to learn multimodal aesthetic representations. Specifically, we pretrain an image-text encoder-decoder model with image-comment pairs, using contrastive and generative objectives to learn rich and generic aesthetic semantics without human labels. To efficiently adapt the pretrained model for downstream IAA tasks, we further propose a lightweight rank-based adapter that employs text as an anchor to learn the aesthetic ranking concept. Our results show that our pretrained aesthetic vision-language model outperforms prior works on image aesthetic captioning over the AVA-Captions dataset, and it has powerful zero-shot capability for aesthetic tasks such as zero-shot style classification and zero-shot IAA, surpassing many supervised baselines. With only minimal finetuning parameters using the proposed adapter module, our model achieves state-of-the-art IAA performance over the AVA dataset. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our model is available at https://github.com/google-research/google-research/tree/master/VILA

        ----

        ## [958] Image Cropping with Spatial-aware Feature and Rank Consistency

        **Authors**: *Chao Wang, Li Niu, Bo Zhang, Liqing Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00969](https://doi.org/10.1109/CVPR52729.2023.00969)

        **Abstract**:

        Image cropping aims to find visually appealing crops in an image. Despite the great progress made by previous methods, they are weak in capturing the spatial relationship between crops and aesthetic elements (e.g., salient objects, semantic edges). Besides, due to the high annotation cost of labeled data, the potential of unlabeled data awaits to be excavated. To address the first issue, we propose spatial-aware feature to encode the spatial relationship between candidate crops and aesthetic elements, by feeding the concatenation of crop mask and selectively aggregated feature maps to a light-weighted encoder. To address the second issue, we train a pair-wise ranking classifier on labeled images and transfer such knowledge to unlabeled images to enforce rank consistency. Experimental results on the benchmark datasets show that our proposed method performs favorably against state-of-the-art methods.

        ----

        ## [959] B-Spline Texture Coefficients Estimator for Screen Content Image Super-Resolution

        **Authors**: *Byeonghyun Pak, Jaewon Lee, Kyong Hwan Jin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00970](https://doi.org/10.1109/CVPR52729.2023.00970)

        **Abstract**:

        Screen content images (SCIs) include many informative components, e.g., texts and graphics. Such content creates sharp edges or homogeneous areas, making a pixel distribution of SCI different from the natural image. Therefore, we need to properly handle the edges and textures to minimize information distortion of the contents when a display device's resolution differs from SCIs. To achieve this goal, we propose an implicit neural representation using B-splines for screen content image super-resolution (SCI SR) with arbitrary scales. Our method extracts scaling, translating, and smoothing parameters of B-splines. The followed multilayer perceptron (MLP) uses the estimated B-splines to recover high-resolution SCI. Our network outperforms both a transformer-based reconstruction and an implicit Fourier representation method in almost upscaling factor, thanks to the positive constraint and compact support of the B-spline basis. Moreover, our SR results are recognized as correct text letters with the highest confidence by a pre-trained scene text recognition network. Source code is available at https://github.com/ByeongHyunPak/btc.

        ----

        ## [960] Delving StyleGAN Inversion for Image Editing: A Foundation Latent Space Viewpoint

        **Authors**: *Hongyu Liu, Yibing Song, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00971](https://doi.org/10.1109/CVPR52729.2023.00971)

        **Abstract**:

        GAN inversion and editing via StyleGAN maps an input image into the embedding spaces (W, W+, and F) to simultaneously maintain image fidelity and meaningful manipulation. From latent space W to extended latent space W+ to feature space F in StyleGAN, the editability of GAN inversion decreases while its reconstruction quality increases. Recent GAN inversion methods typically explore W+ and F rather than W to improve reconstruction fidelity while maintaining editability. As W+ and F are derived from W that is essentially the foundation latent space of StyleGAN, these GAN inversion methods focusing on W+ and F spaces could be improved by stepping back to W. In this work, we propose to first obtain the proper latent code in foundation latent space W. We introduce contrastive learning to align W and the image space for proper latent code discovery. Then, we leverage a cross-attention encoder to transform the obtained latent code in W into W+ and F, accordingly. Our experiments show that our exploration of the foundation latent space W improves the representation ability of latent codes in W+ and features in F, which yields state-of-the-art reconstruction fidelity and editability results on the standard benchmarks. Project page: https://kumapowerliu.github.io/CLCAE.

        ----

        ## [961] Learning Dynamic Style Kernels for Artistic Style Transfer

        **Authors**: *Wenju Xu, Chengjiang Long, Yongwei Nie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00972](https://doi.org/10.1109/CVPR52729.2023.00972)

        **Abstract**:

        Arbitrary style transfer has been demonstrated to be efficient in artistic image generation. Previous methods either globally modulate the content feature ignoring local details, or overly focus on the local structure details leading to style leakage. In contrast to the literature, we propose a new scheme “style kernel” that learns spatially adaptive kernels for per-pixel stylization, where the convolutional kernels are dynamically generated from the global style-content aligned feature and then the learned kernels are applied to modulate the content feature at each spatial position. This new scheme allows flexible both global and local interactions between the content and style features such that the wanted styles can be easily transferred to the content image while at the same time the content structure can be easily preserved. To further enhance the flexibility of our style transfer method, we propose a Style Alignment Encoding (SAE) module complemented with a Content-based Gating Modulation (CGM) module for learning the dynamic style kernels in focusing regions. Extensive experiments strongly demonstrate that our proposed method outperforms state-of-the-art methods and exhibits superior performance in terms of visual quality and efficiency.

        ----

        ## [962] SVGformer: Representation Learning for Continuous Vector Graphics using Transformers

        **Authors**: *Defu Cao, Zhaowen Wang, Jose Echevarria, Yan Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00973](https://doi.org/10.1109/CVPR52729.2023.00973)

        **Abstract**:

        Advances in representation learning have led to great success in understanding and generating data in various domains. However, in modeling vector graphics data, the pure data-driven approach often yields unsatisfactory results in downstream tasks as existing deep learning methods often require the quantization of SVG parameters and cannot exploit the geometric properties explicitly. In this paper, we propose a transformer-based representation learning model (SVG-former) that directly operates on continuous input values and manipulates the geometric information of SVG to encode outline details and long-distance dependencies. SVGfomer can be used for various downstream tasks: reconstruction, classification, interpolation, retrieval, etc. We have conducted extensive experiments on vector font and icon datasets to show that our model can capture high-quality representation information and outperform the previous state-of-the-art on downstream tasks significantly.

        ----

        ## [963] Learning Generative Structure Prior for Blind Text Image Super-resolution

        **Authors**: *Xiaoming Li, Wangmeng Zuo, Chen Change Loy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00974](https://doi.org/10.1109/CVPR52729.2023.00974)

        **Abstract**:

        Blind text image super-resolution (SR) is challenging as one needs to cope with diverse font styles and unknown degradation. To address the problem, existing methods perform character recognition in parallel to regularize the SR task, either through a loss constraint or intermediate feature condition. Nonetheless, the high-level prior could still fail when encountering severe degradation. The prob-lem is further compounded given characters of complex structures, e.g., Chinese characters that combine multiple pictographic or ideographic symbols into a single charac-ter. In this work, we present a novel prior that focuses more on the character structure. In particular, we learn to encapsulate rich and diverse structures in a StyleGAN and exploit such generative structure priors for restoration. To restrict the generative space of StyleGAN so that it obeys the structure of characters yet remains flexible in handling different font styles, we store the discrete features for each character in a codebook. The code subsequently drives the StyleGAN to generate high-resolution structural details to aid text SR. Compared to priors based on character recognition, the proposed structure prior ex-erts stronger character-specific guidance to restore faithful and precise strokes of a designated character. Extensive experiments on synthetic and real datasets demonstrate the compelling performance of the proposed generative structure prior in facilitating robust text SR. Our code is available at https://github.com/csxmli2016/MARCONet.

        ----

        ## [964] Unsupervised Domain Adaption with Pixel-Level Discriminator for Image-Aware Layout Generation

        **Authors**: *Chenchen Xu, Min Zhou, Tiezheng Ge, Yuning Jiang, Weiwei Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00975](https://doi.org/10.1109/CVPR52729.2023.00975)

        **Abstract**:

        Layout is essential for graphic design and poster generation. Recently, applying deep learning models to generate layouts has attracted increasing attention. This paper focuses on using the GAN-based model conditioned on image contents to generate advertising poster graphic layouts, which requires an advertising poster layout dataset with paired product images and graphic layouts. However, the paired images and layouts in the existing dataset are collected by inpainting and annotating posters, respectively. There exists a domain gap between inpainted posters (source domain data) and clean product images (target domain data). Therefore, this paper combines unsupervised domain adaption techniques to design a GAN with a novel pixel-level discriminator (PD), called PDA-GAN, to generate graphic layouts according to image contents. The PD is connected to the shallow level feature map and computes the GAN loss for each input-image pixel. Both quantitative and qualitative evaluations demonstrate that PDAGAN can achieve state-of-the-art performances and generate high-quality image-aware graphic layouts for advertising posters.

        ----

        ## [965] Scaling up GANs for Text-to-Image Synthesis

        **Authors**: *Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, Taesung Park*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00976](https://doi.org/10.1109/CVPR52729.2023.00976)

        **Abstract**:

        The recent success of text-to-image synthesis has taken the world by storm and captured the general public's imagination. From a technical standpoint, it also marked a drastic change in the favored architecture to design generative image models. GANs used to be the de facto choice, with techniques like StyleGAN. With DALL.E 2, autoregressive and diffusion models became the new standard for large-scale generative models overnight. This rapid shift raises a fundamental question: can we scale up GANs to benefit from large datasets like LAION? We find that naïvely increasing the capacity of the StyleGan architecture quickly becomes unstable. We introduce GigaGAN, a new GAN architecture that far exceeds this limit, demonstrating GANs as a viable option for text-to-image synthesis. GigaGAN offers three major advantages. First, it is orders of magnitude faster at inference time, taking only 0.13 seconds to synthesize a 512px image. Second, it can synthesize high-resolution images, for example, 16-megapixel images in 3.66 seconds. Finally, GigaGAN supports various latent space editing applications such as latent interpolation, style mixing, and vector arithmetic operations.

        ----

        ## [966] ERNIE-ViLG 20: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts

        **Authors**: *Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, Yu Sun, Li Chen, Hao Tian, Hua Wu, Haifeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00977](https://doi.org/10.1109/CVPR52729.2023.00977)

        **Abstract**:

        Recent progress in diffusion models has revolutionized the popular technology of text-to-image generation. While existing approaches could produce photorealistic high-resolution images with text conditions, there are still several open problems to be solved, which limits the further improvement of image fidelity and text relevancy. In this paper, we propose ERNIE-ViLG 2.0, a large-scale Chinese text-to-image diffusion model, to progressively upgrade the quality of generated images by: (1) incorporating fine-grained textual and visual knowledge of key elements in the scene, and (2) utilizing different denoising experts at different denoising stages. With the proposed mechanisms, ERNIE-ViLG 2.0
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
 not only achieves a new state-of-the-art on MS-COCO with zero-shot FID-30k score of 6.75, but also significantly outperforms recent models in terms of image fidelity and image-text alignment, with side-by-side human evaluation on the bilingual prompt set ViLG-300.

        ----

        ## [967] Inversion-based Style Transfer with Diffusion Models

        **Authors**: *Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00978](https://doi.org/10.1109/CVPR52729.2023.00978)

        **Abstract**:

        The artistic style within a painting is the means of expression, which includes not only the painting material, colors, and brushstrokes, but also the high-level attributes, including semantic elements and object shapes. Previous arbitrary example-guided artistic image generation methods often fail to control shape changes or convey elements. Pre-trained text-to-image synthesis diffusion probabilistic models have achieved remarkable quality but often require extensive textual descriptions to accurately portray the attributes of a particular painting. The uniqueness of an artwork lies in the fact that it cannot be adequately explained with normal language. Our key idea is to learn the artistic style directly from a single painting and then guide the synthesis without providing complex textual descriptions. Specifically, we perceive style as a learnable textual description of a painting. We propose an inversion-based style transfer method (InST), which can efficiently and accurately learn the key information of an image, thus capturing and transferring the artistic style of a painting. We demonstrate the quality and efficiency of our method on numerous paintings of various artists and styles. Codes are available at https://github.com/zyxElsa/InST.

        ----

        ## [968] Shifted Diffusion for Text-to-image Generation

        **Authors**: *Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, Jinhui Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00979](https://doi.org/10.1109/CVPR52729.2023.00979)

        **Abstract**:

        We present Corgi, a novel method for text-to-image generation. Corgi is based on our proposed shifted diffusion model, which achieves better image embedding generation from input text. Unlike the baseline diffusion model used in DALL-E 2, our method seamlessly encodes prior knowledge of the pre-trained CLIP model in its diffusion process by designing a new initialization distribution and a new transition step of the diffusion. Compared to the strong DALL-E 2 baseline, our method performs better in generating image embedding from the text in terms of both efficiency and effectiveness, resulting in better text-to-image generation. Extensive large-scale experiments are conducted and evaluated in terms of both quantitative measures and human evaluation, indicating a stronger generation ability of our method compared to existing ones. Furthermore, our model enables semi-supervised and language-free training for text-to-image generation, where only part or none of the images in the training dataset have an associated caption. Trained with only 1.7% of the images being captioned, our semi-supervised model obtains FID results comparable to DALL-E 2 on zero-shot text-to-image generation evaluated on MS-COCO. Corgi also achieves new state-of-the-art results across different datasets on downstream language-free text-to-image generation tasks, outperforming the previous method, Lafite, by a large margin.

        ----

        ## [969] LayoutDM: Discrete Diffusion Model for Controllable Layout Generation

        **Authors**: *Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, Kota Yamaguchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00980](https://doi.org/10.1109/CVPR52729.2023.00980)

        **Abstract**:

        Controllable layout generation aims at synthesizing plausible arrangement of element bounding boxes with optional constraints, such as type or position of a specific element. In this work, we try to solve a broad range of layout generation tasks in a single model that is based on discrete state-space diffusion models. Our model, named Lay-outDM, naturally handles the structured layout data in the discrete representation and learns to progressively infer a noiseless layout from the initial input, where we model the layout corruption process by modality-wise discrete diffusion. For conditional generation, we propose to inject layout constraints in the form of masking or logit adjustment during inference. We show in the experiments that our Lay-outDM successfully generates high-quality layouts and outperforms both task-specific and task-agnostic baselines on several layout tasks. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Please find the code and models at: https://cyberagentailab.github.io/layout-drn.

        ----

        ## [970] Unpaired Image-to-Image Translation with Shortest Path Regularization

        **Authors**: *Shaoan Xie, Yanwu Xu, Mingming Gong, Kun Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00981](https://doi.org/10.1109/CVPR52729.2023.00981)

        **Abstract**:

        Unpaired image-to-image translation aims to learn proper mappings that can map images from one domain to another domain while preserving the content of the input image. However, with large enough capacities, the network can learn to map the inputs to any random permutation of images in another domain. Existing methods treat two domains as discrete and propose different assumptions to address this problem. In this paper, we start from a different perspective and consider the paths connecting the two domains. We assume that the optimal path length between the input and output image should be the shortest among all possible paths. Based on this assumption, we propose a new method to allow generating images along the path and present a simple way to encourage the network to find the shortest path without pair information. Extensive experiments on various tasks demonstrate the superiority of our approach. The code is available at https://github.com/Mid-Push/santa.

        ----

        ## [971] DiffCollage: Parallel Generation of Large Content with Diffusion Models

        **Authors**: *Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, Ming-Yu Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00982](https://doi.org/10.1109/CVPR52729.2023.00982)

        **Abstract**:

        We present DiffCollage, a compositional diffusion model that can generate large content by leveraging diffusion models trained on generating pieces of the large content. Our approach is based on a factor graph representation where each factor node represents a portion of the content and a variable node represents their overlap. This representation allows us to aggregate intermediate outputs from diffusion models defined on individual nodes to generate content of arbitrary size and shape in parallel without resorting to an autoregressive generation procedure. We apply DiffCollage to various tasks, including infinite image generation, panorama image generation, and long-duration text-guided motion generation. Extensive experimental results with a comparison to strong autoregressive baselines verify the effectiveness of our approach.

        ----

        ## [972] Wavelet Diffusion Models are fast and scalable Image Generators

        **Authors**: *Hao Phung, Quan Dao, Anh Tran*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00983](https://doi.org/10.1109/CVPR52729.2023.00983)

        **Abstract**:

        Diffusion models are rising as a powerful solution for high-fidelity image generation, which exceeds GANs in quality in many circumstances. However, their slow training and inference speed is a huge bottleneck, blocking them from being used in real-time applications. A recent DiffusionGAN method significantly decreases the models' running time by reducing the number of sampling steps from thousands to several, but their speeds still largely lag behind the GAN counterparts. This paper aims to reduce the speed gap by proposing a novel wavelet-based diffusion scheme. We extract low-and-high frequency components from both image and feature levels via wavelet decomposition and adaptively handle these components for faster processing while maintaining good generation quality. Furthermore, we propose to use a reconstruction term, which effectively boosts the model training convergence. Experimental results on CelebA-HQ, CIFAR-10, LSUN-Church, and STL-10 datasets prove our solution is a stepping-stone to offering real-time and high-fidelity diffusion models. Our code and pre-trained checkpoints are available at https://github.com/VinAIResearch/WaveDiff.git.

        ----

        ## [973] MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation

        **Authors**: *Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, Baining Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00985](https://doi.org/10.1109/CVPR52729.2023.00985)

        **Abstract**:

        We propose the first joint audio-video generation framework that brings engaging watching and listening experiences simultaneously, towards high-quality realistic videos. To generate joint audio-video pairs, we propose a novel Multi-Modal Diffusion model (i.e., MM-Diffusion), with two-coupled denoising autoencoders. In contrast to existing single-modal diffusion models, MM-Diffusion consists of a sequential multi-modal U-Net for a joint denoising process by design. Two subnets for audio and video learn to gradually generate aligned audio-video pairs from Gaussian noises. To ensure semantic consistency across modalities, we propose a novel random-shift based attention block bridging over the two subnets, which enables efficient cross-modal alignment, and thus reinforces the audio-video fidelity for each other. Extensive experiments show superior results in unconditional audio-video generation, and zeroshot conditional tasks (e.g., video-to-audio). In particular, we achieve the best FVD and FAD on Landscape and AIST++ dancing datasets. Turing tests of 10k votes further demonstrate dominant preferences for our model. The code and pre-trained models can be downloaded at https://github.com/researchmm/MM-Diffusion.

        ----

        ## [974] Adaptive Human Matting for Dynamic Videos

        **Authors**: *Chung-Ching Lin, Jiang Wang, Kun Luo, Kevin Lin, Linjie Li, Lijuan Wang, Zicheng Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00986](https://doi.org/10.1109/CVPR52729.2023.00986)

        **Abstract**:

        The most recent efforts in video matting have focused on eliminating trimap dependency since trimap annotations are expensive and trimap-based methods are less adaptable for real-time applications. Despite the latest tripmapfree methods showing promising results, their performance often degrades when dealing with highly diverse and unstructured videos. We address this limitation by introducing Adaptive Matting for Dynamic Videos, termed AdaM, which is a framework designed for simultaneously differentiating foregrounds from backgrounds and capturing alpha matte details of human subjects in the foreground. Two interconnected network designs are employed to achieve this goal: (1) an encoder-decoder network that produces alpha mattes and intermediate masks which are used to guide the transformer in adaptively decoding foregrounds and backgrounds, and (2) a transformer network in which long- and short-term attention combine to retain spatial and temporal contexts, facilitating the decoding of foreground details. We benchmark and study our methods on recently introduced datasets, showing that our model notably improves matting realism and temporal coherence in complex real-world videos and achieves new best-in-class generalizability. Further details and examples are available at https://github.com/microsoft/AdaM.

        ----

        ## [975] LVQAC: Lattice Vector Quantization Coupled with Spatially Adaptive Companding for Efficient Learned Image Compression

        **Authors**: *Xi Zhang, Xiaolin Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00987](https://doi.org/10.1109/CVPR52729.2023.00987)

        **Abstract**:

        Recently, numerous end-to-end optimized image compression neural networks have been developed and proved themselves as leaders in rate-distortion performance. The main strength of these learnt compression methods is in powerful nonlinear analysis and synthesis transforms that can be facilitated by deep neural networks. However, out of operational expediency, most of these end-to-end methods adopt uniform scalar quantizers rather than vector quantizers, which are information-theoretically optimal. In this paper, we present a novel Lattice Vector Quantization scheme coupled with a spatially Adaptive Companding (LVQAC) mapping. LVQ can better exploit the inter-feature dependencies than scalar uniform quantization while being computationally almost as simple as the latter. Moreover, to improve the adaptability of LVQ to source statistics, we couple a spatially adaptive companding (AC) mapping with LVQ. The resulting LVQAC design can be easily embedded into any end-to-end optimized image compression system. Extensive experiments demonstrate that for any end-to-end CNN image compression models, replacing uniform quantiter by LVQAC achieves better rate-distortion performance without significantly increasing the model complexity.

        ----

        ## [976] Hierarchical B-Frame Video Coding Using Two-Layer CANF Without Motion Coding

        **Authors**: *David Alexandre, Hsueh-Ming Hang, Wen-Hsiao Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00988](https://doi.org/10.1109/CVPR52729.2023.00988)

        **Abstract**:

        Typical video compression systems consist of two main modules: motion coding and residual coding. This general architecture is adopted by classical coding schemes (such as international standards H.265 and H.266) and deep learning-based coding schemes. We propose a novel B-frame coding architecture based on two-layer Conditional Augmented Normalization Flows (CANF). It has the striking feature of not transmitting any motion information. Our proposed idea of video compression without motion coding offers a new direction for learned video coding. Our base layer is a low-resolution image compressor that replaces the full-resolution motion compressor. The low-resolution coded image is merged with the warped high-resolution images to generate a high-quality image as a conditioning signal for the enhancement-layer image coding in full resolution. One advantage of this architecture is significantly reduced computational complexity due to eliminating the motion information compressor. In addition, we adopt a skip-mode coding technique to reduce the transmitted latent samples. The rate-distortion performance of our scheme is slightly lower than that of the state-of-the-art learned B-frame coding scheme, B-CANF, but outperforms other learned B-frame coding schemes. However, compared to B-CANF, our scheme saves 45% of multiply-accumulate operations (MACs) for encoding and 27% of MACs for decoding. The code is available at https://nycu-clab.github.io.

        ----

        ## [977] Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting

        **Authors**: *Gen Li, Jie Ji, Minghai Qin, Wei Niu, Bin Ren, Fatemeh Afghah, Linke Guo, Xiaolong Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00989](https://doi.org/10.1109/CVPR52729.2023.00989)

        **Abstract**:

        As deep convolutional neural networks (DNNs) are widely used in various fields of computer vision, leveraging the overfitting ability of the DNN to achieve video resolution upscaling has become a new trend in the modern video delivery system. By dividing videos into chunks and over-fitting each chunk with a super-resolution model, the server encodes videos before transmitting them to the clients, thus achieving better video quality and transmission efficiency. However, a large number of chunks are expected to ensure good overfitting quality, which substantially increases the storage and consumes more bandwidth resources for data transmission. On the other hand, decreasing the number of chunks through training optimization techniques usually requires high model capacity, which significantly slows down execution speed. To reconcile such, we propose a novel method for high-quality and efficient video resolution upscaling tasks, which leverages the spatial-temporal information to accurately divide video into chunks, thus keeping the number of chunks as well as the model size to minimum. Additionally, we advance our method into a single overfitting model by a data-aware joint training technique. which further reduces the storage requirement with negligible quality drop. We deploy our models on an off-the-shelf mobile phone, and experimental results show that our method achieves real-time video super-resolution with high video quality. Compared with the state-of-the-art, our method achieves 28 fps streaming speed with 41.6 PSNR, which is 14 × faster and 2.29 dB better in the live video resolution upscaling tasks. Code available in https://github.com/coulsonlee/STDO-CVPR2023.git.

        ----

        ## [978] HNeRV: A Hybrid Neural Representation for Videos

        **Authors**: *Hao Chen, Matthew Gwilliam, Ser-Nam Lim, Abhinav Shrivastava*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00990](https://doi.org/10.1109/CVPR52729.2023.00990)

        **Abstract**:

        Implicit neural representations store videos as neural networks and have performed well for various vision tasks such as video compression and denoising. With frame index or positional index as input, implicit representations (NeRV, E-NeRV, etc.) reconstruct video frames from fixed and content-agnostic embeddings. Such embedding largely limits the regression capacity and internal generalization for video interpolation. In this paper, we propose a Hybrid Neural Representation for Videos (HNeRV), where a learnable encoder generates content-adaptive embeddings, which act as the decoder input. Besides the input embedding, we introduce HNeRV blocks, which ensure model parameters are evenly distributed across the entire network, such that higher layers (layers near the output) can have more capacity to store high-resolution content and video details. With content-adaptive embeddings and redesigned architecture, HNeRV outperforms implicit methods in video regression tasks for both reconstruction quality (+4.7 PSNR) and convergence speed (16 × faster), and shows better internal generalization. As a simple and efficient video representation, HNeRV also shows decoding advantages for speed, flexibility, and deployment, compared to traditional codecs (H.264, H.265) and learning-based compression methods. Finally, we explore the effectiveness of HNeRV on downstream tasks such as video compression and video inpainting.

        ----

        ## [979] Regularize implicit neural representation by itself

        **Authors**: *Zhemin Li, Hongxia Wang, Deyu Meng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00991](https://doi.org/10.1109/CVPR52729.2023.00991)

        **Abstract**:

        This paper proposes a regularizer called Implicit Neural Representation Regularizer (INRR) to improve the generalization ability of the Implicit Neural Representation (INR). The INR is a fully connected network that can represent signals with details not restricted by grid resolution. However, its generalization ability could be improved, especially with nonuniformly sampled data. The proposed INRR is based on learned Dirichlet Energy (DE) that measures similarities between rows/columns of the matrix. The smoothness of the Laplacian matrix is further integrated by parameterizing DE with a tiny INR. INRR improves the generalization of INR in signal representation by perfectly integrating the signal's self-similarity with the smoothness of the Laplacian matrix. Through well-designed numerical experiments, the paper also reveals a series of properties derived from INRR, including momentum methods like convergence trajectory and multi-scale similarity. Moreover, the proposed method could improve the performance of other signal representation methods.

        ----

        ## [980] SMPConv: Self-Moving Point Representations for Continuous Convolution

        **Authors**: *Sanghyeon Kim, Eunbyung Park*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00992](https://doi.org/10.1109/CVPR52729.2023.00992)

        **Abstract**:

        Continuous convolution has recently gained prominence due to its ability to handle irregularly sampled data and model long-term dependency. Also, the promising experimental results of using large convolutional kernels have catalyzed the development of continuous convolution since they can construct large kernels very efficiently. Lever-aging neural networks, more specifically multilayer perceptrons (MLPs), is by far the most prevalent approach to implementing continuous convolution. However, there are a few drawbacks, such as high computational costs, complex hyperparameter tuning, and limited descriptive power of filters. This paper suggests an alternative approach to building a continuous convolution without neural networks, resulting in more computationally efficient and improved performance. We present self-moving point representations where weight parameters freely move, and interpolation schemes are used to implement continuous functions. When applied to construct convolutional kernels, the experimental results have shown improved performance with drop-in replacement in the existing frame-works. Due to its lightweight structure, we are first to demonstrate the effectiveness of continuous convolution in a large-scale setting, e.g., ImageNet, presenting the improvements over the prior arts. Our code is available on https://github.com/sangnekim/SMPConv

        ----

        ## [981] Long Range Pooling for 3D Large-Scale Scene Understanding

        **Authors**: *Xiang-Li Li, Meng-Hao Guo, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00993](https://doi.org/10.1109/CVPR52729.2023.00993)

        **Abstract**:

        Inspired by the success of recent vision transformers and large kernel design in convolutional neural networks (CNNs), in this paper, we analyze and explore essential reasons for their success. We claim two factors that are critical for 3D large-scale scene understanding: a larger receptive field and operations with greater non-linearity. The former is responsible for providing long range contexts and the latter can enhance the capacity of the network. To achieve the above properties, we propose a simple yet effective long range pooling (LRP) module using dilation max pooling, which provides a network with a large adaptive receptive field. LRP has few parameters, and can be readily added to current CNNs. Also, based on LRP, we present an entire network architecture, LRPNet, for 3D understanding. Ablation studies are presented to support our claims, and show that the LRP module achieves better results than large kernel convolution yet with reduced computation, due to its non-linearity. We also demonstrate the superiority of LRPNet on various benchmarks: LRPNet performs the best on ScanNet and surpasses other CNN-based methods on S3DIS and Matterport3D. Code will be avalible at https://github.com/li-xl/LRPNet.

        ----

        ## [982] Progressive Random Convolutions for Single Domain Generalization

        **Authors**: *Seokeon Choi, Debasmit Das, Sungha Choi, Seunghan Yang, Hyunsin Park, Sungrack Yun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00994](https://doi.org/10.1109/CVPR52729.2023.00994)

        **Abstract**:

        Single domain generalization aims to train a generalizable model with only one source domain to perform well on arbitrary unseen target domains. Image augmentation based on Random Convolutions (RandConv), consisting of one convolution layer randomly initialized for each mini-batch, enables the model to learn generalizable visual representations by distorting local textures despite its simple and lightweight structure. However, RandConv has structural limitations in that the generated image easily loses semantics as the kernel size increases, and lacks the inherent diversity of a single convolution operation. To solve the problem, we propose a Progressive Random Convolution (Pro-RandConv) method that recursively stacks random convolution layers with a small kernel size instead of increasing the kernel size. This progressive approach can not only mitigate semantic distortions by reducing the influence of pixels away from the center in the theoretical receptive field, but also create more effective virtual domains by gradually increasing the style diversity. In addition, we develop a basic random convolution layer into a random convolution block including deformable offsets and affine transformation to support texture and contrast diversification, both of which are also randomly initialized. Without complex generators or adversarial learning, we demonstrate that our simple yet effective augmentation strategy outperforms state-of-the-art methods on single domain generalization benchmarks.

        ----

        ## [983] BiFormer: Vision Transformer with Bi-Level Routing Attention

        **Authors**: *Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, Rynson W. H. Lau*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00995](https://doi.org/10.1109/CVPR52729.2023.00995)

        **Abstract**:

        As the core building block of vision transformers, attention is a powerful tool to capture long-range dependency. However, such power comes at a cost: it incurs a huge computation burden and heavy memory footprint as pairwise token interaction across all spatial locations is computed. A series of works attempt to alleviate this problem by introducing handcrafted and content-agnostic sparsity into attention, such as restricting the attention operation to be inside local windows, axial stripes, or dilated windows. In contrast to these approaches, we propose a novel dynamic sparse attention via bi-level routing to enable a more flexible allocation of computations with content awareness. Specifically, for a query, irrelevant key-value pairs are first filtered out at a coarse region level, and then fine-grained token-to-token attention is applied in the union of remaining candidate regions (i.e., routed regions). We provide a simple yet effective implementation of the proposed bilevel routing attention, which utilizes the sparsity to save both computation and memory while involving only GPU-friendly dense matrix multiplications. Built with the proposed bi-level routing attention, a new general vision transformer, named BiFormer, is then presented. As BiFormer attends to a small subset of relevant tokens in a query adaptive manner without distraction from other irrelevant ones, it enjoys both good performance and high computational efficiency, especially in dense prediction tasks. Empirical results across several computer vision tasks such as image classification, object detection, and semantic segmentation verify the effectiveness of our design. Code is available at https://github.com/rayleizhu/BiFormer.

        ----

        ## [984] Beyond Attentive Tokens: Incorporating Token Importance and Diversity for Efficient Vision Transformers

        **Authors**: *Sifan Long, Zhen Zhao, Jimin Pi, Shengsheng Wang, Jingdong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00996](https://doi.org/10.1109/CVPR52729.2023.00996)

        **Abstract**:

        Vision transformers have achieved significant improvements on various vision tasks but their quadratic interactions between tokens significantly reduce computational efficiency. Many pruning methods have been proposed to remove redundant tokens for efficient vision transformers recently. However, existing studies mainly focus on the token importance to preserve local attentive tokens but completely ignore the global token diversity. In this paper, we emphasize the cruciality of diverse global semantics and propose an efficient token decoupling and merging method that can jointly consider the token importance and diversity for token pruning. According to the class token attention, we decouple the attentive and inattentive tokens. In addition to preserving the most discriminative local tokens, we merge similar inattentive tokens and match homogeneous attentive tokens to maximize the token diversity. Despite its simplicity, our method obtains a promising trade-off between model complexity and classification accuracy. On DeiT-S, our method reduces the FLOPs by 35% with only a 0.2% accuracy drop. Notably, benefiting from maintaining the token diversity, our method can even improve the accuracy of DeiT-T by 0.1% after reducing its FLOPs by 40%.

        ----

        ## [985] BioNet: A Biologically-Inspired Network for Face Recognition

        **Authors**: *Pengyu Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00997](https://doi.org/10.1109/CVPR52729.2023.00997)

        **Abstract**:

        Recently, whether and how cutting-edge Neuroscience findings can inspire Artificial Intelligence (AI) confuse both communities and draw much discussion. As one of the most critical fields in AI, Computer Vision (CV) also pays much attention to the discussion. To show our ideas and experimental evidence to the discussion, we focus on one of the most broadly researched topics both in Neuroscience and CV fields, i.e., Face Recognition (FR). Neuroscience studies show that face attributes are essential to the human face-recognizing system. How the attributes contribute also be explained by the Neuroscience community. Even though a few CV works improved the FR performance with attribute enhancement, none of them are inspired by the human face-recognizing mechanism nor boosted performance significantly. To show our idea experimentally, we model the biological characteristics of the human face-recognizing system with classical Convolutional Neural Network Operators (CNN Ops) purposely. We name the proposed Biologically-inspired Network as BioNet. Our BioNet consists of two cascade sub-networks, i.e., the Visual Cortex Network (VCN) and the Inferotemporal Cortex Network (ICN). The VCN is modeled with a classical CNN backbone. The proposed ICN comprises three biologically-inspired modules, i.e., the Cortex Functional Compartmentalization, the Compartment Response Transform, and the Response Intensity Modulation. The experiments prove that: 1) The cutting-edge findings about the human face-recognizing system can further boost the CNN-based FR network. 2) With the biological mechanism, both identity-related attributes (e.g., gender) and identity-unrelated attributes (e.g., expression) can benefit the deep FR models. Surprisingly, the identity-unrelated ones contribute even more than the identity-related ones. 3) The proposed BioNet significantly boosts state-of-the-art on standard FR benchmark datasets. For example, BioNet boosts IJB-B@ 1 e-6 from 52.12% to 68.28% and MegaFace from 98.74% to 99.19%. The source code is released in
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/pengyuLPY/BioNet.git.

        ----

        ## [986] Dual-bridging with Adversarial Noise Generation for Domain Adaptive rPPG Estimation

        **Authors**: *Jingda Du, Siqi Liu, Bochao Zhang, Pong C. Yuen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00998](https://doi.org/10.1109/CVPR52729.2023.00998)

        **Abstract**:

        The remote photoplethysmography (rPPG) technique can estimate pulse-related metrics (e.g. heart rate and respiratory rate) from facial videos and has a high potential for health monitoring. The latest deep rPPG methods can model in-distribution noise due to head motion, video compression, etc., and estimate high-quality rPPG signals under similar scenarios. However, deep rPPG models may not generalize well to the target test domain with unseen noise and distortions. In this paper, to improve the generalization ability of rPPG models, we propose a dual-bridging network to reduce the domain discrepancy by aligning intermediate domains and synthesizing the target noise in the source domain for better noise reduction. To comprehensively explore the target domain noise, we propose a novel adversarial noise generation in which the noise generator indirectly competes with the noise reducer. To further improve the robustness of the noise reducer, we propose hard noise pattern mining to encourage the generator to learn hard noise patterns contained in the target domain features. We evaluated the proposed method on three public datasets with different types of interferences. Under different crossdomain scenarios, the comprehensive results show the effectiveness of our method.

        ----

        ## [987] On Data Scaling in Masked Image Modeling

        **Authors**: *Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Yixuan Wei, Qi Dai, Han Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00999](https://doi.org/10.1109/CVPR52729.2023.00999)

        **Abstract**:

        Scaling properties have been one of the central issues in self-supervised pre-training, especially the data scalability, which has successfully motivated the large-scale self-supervised pre-trained language models and endowed them with significant modeling capabilities. However, scaling properties seem to be unintentionally neglected in the recent trending studies on masked image modeling (MIM), and some arguments even suggest that MIM cannot benefit from large-scale data. In this work, we try to break down these preconceptions and systematically study the scaling behaviors of MIM through extensive experiments, with data ranging from 10% of ImageNet-1K to full ImageNet-22K, model parameters ranging from 49-million to one-billion, and training length ranging from 125K to 500K iterations. And our main findings can be summarized in two folds: 1) masked image modeling remains demanding large-scale data in order to scale up computes and model parameters; 2) masked image modeling cannot benefit from more data under a non-overfitting scenario, which diverges from the previous observations in self-supervised pre-trained language models or supervised pre-trained vision models. In addition, we reveal several intriguing properties in MIM, such as high sample efficiency in large MIM models and strong correlation between pre-training validation loss and transfer performance. We hope that our findings could deepen the understanding of masked image modeling and facilitate future developments on largescale vision models. Code and models will be available at https://github.com/microsoft/SimMIM.

        ----

        ## [988] Hard Patches Mining for Masked Image Modeling

        **Authors**: *Haochen Wang, Kaiyou Song, Junsong Fan, Yuxi Wang, Jin Xie, Zhaoxiang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01000](https://doi.org/10.1109/CVPR52729.2023.01000)

        **Abstract**:

        Masked image modeling (MIM) has attracted much research attention due to its promising potential for learning scalable visual representations. In typical approaches, models usually focus on predicting specific contents of masked patches, and their performances are highly related to pre-defined mask strategies. Intuitively, this procedure can be considered as training a student (the model) on solving given problems (predict masked patches). However, we argue that the model should not only focus on solving given problems, but also stand in the shoes of a teacher to produce a more challenging problem by itself. To this end, we propose Hard Patches Mining (HPM), a brand-new framework for MIM pre-training. We observe that the reconstruction loss can naturally be the metric of the difficulty of the pretraining task. Therefore, we introduce an auxiliary loss predictor, predicting patch-wise losses first and deciding where to mask next. It adopts a relative relationship learning strategy to prevent overfitting to exact reconstruction loss values. Experiments under various settings demonstrate the effectiveness of HPM in constructing masked images. Furthermore, we empirically find that solely introducing the loss prediction objective leads to powerful representations, verifying the efficacy of the ability to be aware of where is hard to reconstruct.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/Haochen-wang409/HPM

        ----

        ## [989] Evolved Part Masking for Self-Supervised Learning

        **Authors**: *Zhanzhou Feng, Shiliang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01001](https://doi.org/10.1109/CVPR52729.2023.01001)

        **Abstract**:

        Existing Masked Image Modeling methods apply fixed mask patterns to guide the self-supervised training. As those patterns resort to different criteria to mask local regions, sticking to a fixed pattern leads to limited vision cues modeling capability. This paper proposes an evolved part-based masking to pursue more general visual cues modeling in self-supervised learning. Our method is based on an adaptive part partition module, which leverages the vision model being trained to construct a part graph, and partitions parts with graph cut. The accuracy of partitioned parts is on par with the capability of the pretrained model, leading to evolved mask patterns at different training stages. It generates simple patterns at the initial training stage to learn low-level visual cues, which hence evolves to eliminate accurate object parts to reinforce the learning of object semantics and contexts. Our method does not require extra pretrained models or annotations, and effectively ensures the training efficiency by evolving the training difficulty. Experiment results show that it substantially boosts the performance on various tasks including image classification, object detection, and semantic segmentation. For example, it outperforms the recent MAE by 0.69% on imageNet-1K classification and 1.61% on ADE20K segmentation with the same training epochs.

        ----

        ## [990] BASiS: Batch Aligned Spectral Embedding Space

        **Authors**: *Or Streicher, Ido Cohen, Guy Gilboa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01002](https://doi.org/10.1109/CVPR52729.2023.01002)

        **Abstract**:

        Graph is a highly generic and diverse representation, suitable for almost any data processing problem. Spectral graph theory has been shown to provide powerful algorithms, backed by solid linear algebra theory. It thus can be extremely instrumental to design deep network building blocks with spectral graph characteristics. For instance, such a network allows the design of optimal graphs for certain tasks or obtaining a canonical orthogonal low-dimensional embedding of the data. Recent attempts to solve this problem were based on minimizing Rayleigh-quotient type losses. We propose a different approach of directly learning the graph's eigensapce. A severe problem of the direct approach, applied in batch-learning, is the inconsistent mapping of features to eigenspace coordinates in different batches. We analyze the degrees of freedom of learning this task using batches and propose a stable alignment mechanism that can work both with batch changes and with graph-metric changes. We show that our learnt spectral embedding is better in terms of NMI, ACC, Grassman distnace, orthogonality and classification accuracy, compared to SOTA. In addition, the learning is more stable.

        ----

        ## [991] OmniMAE: Single Model Masked Pretraining on Images and Videos

        **Authors**: *Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01003](https://doi.org/10.1109/CVPR52729.2023.01003)

        **Abstract**:

        Transformer-based architectures have become competitive across a variety of visual domains, most notably images and videos. While prior work studies these modalities in isolation, having a common architecture suggests that one can train a single unified model for multiple visual modalities. Prior attempts at unified modeling typically use architectures tailored for vision tasks, or obtain worse performance compared to single modality models. In this work, we show that masked autoencoding can be used to train a simple Vision Transformer on images and videos, without requiring any labeled data. This single model learns visual representations that are comparable to or better than single-modality representations on both image and video benchmarks, while using a much simpler architecture. Furthermore, this model can be learned by dropping 90% of the image and 95% of the video patches, enabling extremely fast training of huge model architectures. In particular, we show that our single ViT-Huge model can be finetuned to achieve 86.6% on ImageNet and 75.5% on the challenging Something Something-v2 video benchmark, setting a new state-of-the-art.

        ----

        ## [992] ViTs for SITS: Vision Transformers for Satellite Image Time Series

        **Authors**: *Michail Tarasiou, Erik Chavez, Stefanos Zafeiriou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01004](https://doi.org/10.1109/CVPR52729.2023.01004)

        **Abstract**:

        In this paper we introduce the Temporo-Spatial Vision Transformer (TSViT), a fully-attentional model for general Satellite Image Time Series (SITS) processing based on the Vision Transformer (ViT). TSViT splits a SITS record into non-overlapping patches in space and time which are tokenized and subsequently processed by a factorized temporo-spatial encoder. We argue, that in contrast to natural images, a temporal-then-spatial factorization is more intuitive for SITS processing and present experimental evidence for this claim. Additionally, we enhance the model's discriminative power by introducing two novel mechanisms for acquisition-time-specific temporal positional encodings and multiple learnable class tokens. The effect of all novel design choices is evaluated through an extensive ablation study. Our proposed architecture achieves state-of-the-art performance, surpassing previous approaches by a significant margin in three publicly available SITS semantic segmentation and classification datasets. All model, training and evaluation codes can be found at https://github.com/michaeltrs/DeepSatModels.

        ----

        ## [993] Probabilistic Debiasing of Scene Graphs

        **Authors**: *Bashirul Azam Biswas, Qiang Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01005](https://doi.org/10.1109/CVPR52729.2023.01005)

        **Abstract**:

        The quality of scene graphs generated by the state-of-the-art (SOTA) models is compromised due to the long-tail nature of the relationships and their parent object pairs. Training of the scene graphs is dominated by the majority relationships of the majority pairs and, therefore, the object-conditional distributions of relationship in the minority pairs are not preserved after the training is converged. Consequently, the biased model performs well on more frequent relationships in the marginal distribution of relationships such as ‘on’ and 'wearing’, and performs poorly on the less frequent relationships such as ‘eating’ or ‘hanging from’. In this work, we propose virtual evidence incorporated within-triplet Bayesian Network (BN) to preserve the object-conditional distribution of the relationship label and to eradicate the bias created by the marginal probability of the relationships. The insufficient number of relationships in the minority classes poses a significant problem in learning the within-triplet Bayesian network. We address this insufficiency by embedding-based augmentation of triplets where we borrow samples of the minority triplet classes from its neighboring triplets in the semantic space. We perform experiments on two different datasets and achieve a significant improvement in the mean recall of the relationships. We also achieve a better balance between recall and mean recall performance compared to the SOTA de-biasing techniques of scene graph models
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
.

        ----

        ## [994] Blind Video Deflickering by Neural Filtering with a Flawed Atlas

        **Authors**: *Chenyang Lei, Xuanchi Ren, Zhaoxiang Zhang, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01006](https://doi.org/10.1109/CVPR52729.2023.01006)

        **Abstract**:

        Many videos contain flickering artifacts; common causes of flicker include video processing algorithms, video generation algorithms, and capturing videos under specific situations. Prior work usually requires specific guidance such as the flickering frequency, manual annotations, or extra consistent videos to remove the flicker. In this work, we propose a general flicker removal framework that only receives a single flickering video as input without additional guidance. Since it is blind to a specific flickering type or guidance, we name this “blind deflickering.” The core of our approach is utilizing the neural atlas in cooperation with a neural filtering strategy. The neural atlas is a unified representation for all frames in a video that provides temporal consistency guidance but is flawed in many cases. To this end, a neural network is trained to mimic a filter to learn the consistent features (e.g., color, brightness) and avoid introducing the artifacts in the atlas. To validate our method, we construct a dataset that contains diverse real-world flickering videos. Extensive experiments show that our method achieves satisfying deflickering performance and even outperforms baselines that use extra guidance on a public benchmark. The source code is publicly available at https://chenyanglei.github.io/deflicker.

        ----

        ## [995] SCOTCH and SODA: A Transformer Video Shadow Detection Framework

        **Authors**: *Lihao Liu, Jean Prost, Lei Zhu, Nicolas Papadakis, Pietro Liò, Carola-Bibiane Schönlieb, Angelica I. Avilés-Rivero*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01007](https://doi.org/10.1109/CVPR52729.2023.01007)

        **Abstract**:

        Shadows in videos are difficult to detect because of the large shadow deformation between frames. In this work, we argue that accounting for shadow deformation is essential when designing a video shadow detection method. To this end, we introduce the shadow deformation attention trajectory (SODA), a new type of video self-attention module, specially designed to handle the large shadow deformations in videos. Moreover, we present a new shadow contrastive learning mechanism (SCOTCH) which aims at guiding the network to learn a unified shadow representation from massive positive shadow pairs across different videos. We demonstrate empirically the effectiveness of our two contributions in an ablation study. Furthermore, we show that SCOTCH and SODA significantly outperforms existing techniques for video shadow detection. Code is available at the project page: https://lihaoliu-cambridge.github.io/scotch_and_soda/

        ----

        ## [996] MAGVIT: Masked Generative Video Transformer

        **Authors**: *Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01008](https://doi.org/10.1109/CVPR52729.2023.01008)

        **Abstract**:

        We introduce the MAsked Generative VIdeo Transformer, MAGVIT, to tackle various video synthesis tasks with a single model. We introduce a 3D tokenizer to quantize a video into spatial-temporal visual tokens and propose an embedding method for masked video token modeling to facilitate multi-task learning. We conduct extensive experiments to demonstrate the quality, efficiency, and flexibility of MAGVIT. Our experiments show that (i) MAGVIT performs favorably against state-of-the-art approaches and establishes the best-published FVD on three video generation benchmarks, including the challenging Kinetics-600. (ii) MAGVIT outperforms existing methods in inference time by two orders of magnitude against diffusion models and by 60x against autoregressive models. (iii) A single MAGVIT model supports ten diverse generation tasks and generalizes across videos from different visual domains. The source code and trained models will be released to the public at https://magvit.cs.cmu.edu.

        ----

        ## [997] Improving Robustness of Semantic Segmentation to Motion-Blur Using Class-Centric Augmentation

        **Authors**: *Aakanksha, A. N. Rajagopalan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01009](https://doi.org/10.1109/CVPR52729.2023.01009)

        **Abstract**:

        Semantic segmentation involves classifying each pixel into one of a pre-defined set of object/stuff classes. Such a fine-grained detection and localization of objects in the scene is challenging by itself. The complexity increases manifold in the presence of blur. With cameras becoming increasingly light-weight and compact, blur caused by motion during capture time has become unavoidable. Most research has focused on improving segmentation performance for sharp clean images and the few works that deal with degradations, consider motion-blur as one of many generic degradations. In this work, we focus exclusively on motion-blur and attempt to achieve robustness for semantic segmentation in its presence. Based on the observation that segmentation annotations can be used to generate synthetic space-variant blur, we propose a Class-Centric Motion-Blur Augmentation (CCMBA) strategy. Our approach involves randomly selecting a subset of semantic classes present in the image and using the segmentation map annotations to blur only the corresponding regions. This enables the network to simultaneously learn semantic segmentation for clean images, images with egomotion blur, as well as images with dynamic scene blur. We demonstrate the effectiveness of our approach for both CNN and Vision Transformer-based semantic segmentation networks on PASCAL VOC and Cityscapes datasets. We also illustrate the improved generalizability of our method to complex real-world blur by evaluating on the commonly used deblurring datasets GoPro and REDS.

        ----

        ## [998] MobileVOS: Real-Time Video Object Segmentation Contrastive Learning meets Knowledge Distillation

        **Authors**: *Roy Miles, Mehmet Kerim Yucel, Bruno Manganelli, Albert Saà-Garriga*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01010](https://doi.org/10.1109/CVPR52729.2023.01010)

        **Abstract**:

        This paper tackles the problem of semi-supervised video object segmentation on resource-constrained devices, such as mobile phones. We formulate this problem as a distillation task, whereby we demonstrate that small space-time-memory networks with finite memory can achieve competitive results with state of the art, but at a fraction of the computational cost (32 milliseconds per frame on a Samsung Galaxy S22). Specifically, we provide a theoretically grounded framework that unifies knowledge distillation with supervised contrastive representation learning. These models are able to jointly benefit from both pixel-wise contrastive learning and distillation from a pre-trained teacher. We validate this loss by achieving competitive J &F to state of the art on both the standard DAVIS and YouTube benchmarks, despite running up to × 5 faster, and with × 32 fewer parameters.

        ----

        ## [999] Self-Supervised Video Forensics by Audio-Visual Anomaly Detection

        **Authors**: *Chao Feng, Ziyang Chen, Andrew Owens*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01011](https://doi.org/10.1109/CVPR52729.2023.01011)

        **Abstract**:

        Manipulated videos often contain subtle inconsistencies between their visual and audio signals. We propose a video forensics method, based on anomaly detection, that can identify these inconsistencies, and that can be trained solely using real, unlabeled data. We train an autoregressive model to generate sequences of audio-visual features, using feature sets that capture the temporal synchronization between video frames and sound. At test time, we then flag videos that the model assigns low probability. Despite being trained entirely on real videos, our model obtains strong performance on the task of detecting manipulated speech videos. Project site: https://cfeng16.github.io/audio-visual-forensics.

        ----

        

[Go to the previous page](CVPR-2023-list04.md)

[Go to the next page](CVPR-2023-list06.md)

[Go to the catalog section](README.md)