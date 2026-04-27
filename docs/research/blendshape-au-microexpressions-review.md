# Blendshapes, Facial Action Units, and Microexpressions: A Focused Literature Review
## 1. Overview and Scope
This review surveys how contemporary literature connects three layers of facial expression representation: (i) the Facial Action Coding System (FACS) and its Action Units (AUs), (ii) blendshape-based 3D facial animation, and (iii) microexpression modeling and recognition.
The focus is on work that either explicitly aligns blendshapes with AUs, or uses AU time-series as a bridge between psychological models of facial expressions (including microexpressions) and practical animation or recognition systems.[^1][^2][^3]
It emphasizes recent developments in AU–blendshape datasets and mappings, AU-based microexpression analysis, and AU-driven animation of micro and macro expressions.
## 2. Background: FACS, AUs, and Microexpressions
The Facial Action Coding System (FACS), introduced by Ekman and colleagues, decomposes facial movement into Action Units, each corresponding to a largely independent contraction of one or more facial muscles.[^4]
AUs are combined to describe complex expressions, with intensities and temporal segments (onset, apex, offset) providing a fine-grained representation of facial dynamics.[^4]

Microexpressions are defined as very brief, subtle facial expressions that occur when individuals attempt to conceal or inhibit their true emotions, typically lasting around 0.05–0.2 seconds, much shorter than conventional (macro) expressions.[^5][^6]
Survey work in psychology and computer vision emphasizes that microexpressions involve the same underlying muscular actions as regular expressions, but reduced in duration and often amplitude.[^6][^7]
Modern overviews distinguish four main microexpression tasks: spotting (temporal localization), recognition (emotion classification), AU detection, and microexpression generation.[^7]

Automatic microexpression analysis research commonly relies on FACS-coded AU ground truth, supporting the view that AUs serve as the core latent representation even when the end goal is emotion classification.[^8][^6]
This AU-centric perspective is important for connecting microexpression literature to blendshape-based animation pipelines, where AUs can be used as an intermediate control space.
## 3. Blendshape-Based Facial Animation and AU Alignment
Blendshapes (or morph targets) are a standard parametric representation for facial animation, modeling deformations of a neutral mesh as a set of basis shapes that can be linearly combined via scalar weights.[^2]
Their popularity arises from implementation simplicity, compatibility with real-time engines, and ease of artistic control compared with more complex muscle or physics-based models.[^2]

Recent work explicitly treats blendshapes as a representation of Action Units.
For example, a preprint for IEEE VR states that a practical face representation can be built where each blendshape “represents a modeled single action unit of the face with respect to the neutral face,” positioning blendshapes as direct geometric realizations of AUs.[^9]
Similarly, Microsoft’s HeadBox toolkit reports creating dozens of FACS-inspired blendshapes as part of an avatar library, explicitly selecting visemes, AUs, and custom expressions to ensure FACS-standardized facial motion across characters.[^10]

In addition, commercial and open-source systems often define blendshape sets that correspond qualitatively to FACS components.
Apple’s ARKit, for example, provides a fixed list of facial blendshape coefficients (such as `browInnerUp`, `browOuterUpLeft`, `mouthSmileLeft`) that can be mapped to specific AUs; practitioners have published “ARKit to FACS” cheat sheets enumerating approximate mappings between ARKit coefficients and AUs.[^11]
These mappings enable AU-based analysis (or capture systems) to drive ARKit-style rigs without manual re-rigging.
## 4. AU–Blendshape Datasets and Models
### 4.1 AUBlendSet and AUBlendNet
The AU-Blendshape line of work introduces a 3D dataset and models designed explicitly around AU–blendshape correspondence.[^3][^12]
AUBlendSet is a blendshape data collection constructed from 32 standard facial AUs across 500 identities, with each identity providing AU-aligned deformations and additional facial postures annotated with detailed AUs.[^3]
This yields a large corpus of AU-specific blendshapes suitable for learning AU-conditioned expression bases.

On top of AUBlendSet, AUBlendNet is proposed as a network that predicts AU-blendshape basis vectors for different character styles: given an identity mesh, it outputs a set of AU-aligned basis shapes that match both the AUs and the stylistic constraints of the target character.[^12][^3]
Applications include stylized facial expression manipulation, speech-driven emotional animation, and data augmentation for emotion recognition, demonstrating that an AU–blendshape representation is usable as a universal control space that bridges data-driven manipulation, animation, and recognition.[^3]

The supplementary material of AU-Blendshape describes a user control interface where operators can freely combine AUs, adjust activation strengths, and compare AUBlendNet’s AU-driven manipulation with ARKit-based and other emotion control schemes.[^12]
This interface illustrates the practical value of treating AUs as the semantic control layer and blendshapes as the geometric actuation layer.
### 4.2 AU-aware 3D Face Reconstruction
Beyond animation, AU-aligned blendshapes have been used in 3D face reconstruction.
An ECCV 2022 paper on AU-aware 3D face reconstruction defines subject-specific AU-blendshapes and linearly combines them with expression coefficients to describe the final 3D face shape, essentially embedding AUs into the reconstruction model.[^13]
This further supports the conceptualization of blendshapes as AU basis functions and shows how AU parameters can be directly integrated into geometric models.
### 4.3 AU-Blendshape in VR and Avatar Control
A preprint aimed at IEEE VR further argues that blendshapes provide a practical and efficient representation for controlling avatar facial expressions and advocates modeling each blendshape as a single AU relative to neutral.[^9]
This offers a clear design principle for avatar rigs intended to be driven by AU-based signals (for example, from AU detectors or microexpression recognition systems).

Taken together, these works establish a trend toward explicitly AU-structured blendshape spaces, in contrast with older rigs where blendshapes were largely ad hoc or expression-level shapes.
## 5. Mapping AUs to Blendshapes
### 5.1 Facial Action Units to Avatars (Wolff et al.)
The paper “Mapping of Facial Action Units to Virtual Avatar Blend Shape Movement” (also distributed under the title “Facial Action Units to Avatars”) directly addresses the problem of mapping between AU intensities and blendshape weights.[^1][^2]
Recognizing that AUs and blendshapes are two different frameworks for describing facial movement, the authors propose a machine learning approach to infer ARKit-compatible blendshape weights from AUs extracted with OpenFace.[^1]

Their model uses a GRU-based recurrent neural network to retain temporal information while achieving fast, real-time inference, and the reported generalized model yields an activation precision of about 90% and recall of about 85% for blendshape activation.[^1]
The system demonstrates that AU time-series can be reliably converted into blendshape controls, thereby leveraging the extensive FACS literature for real-time avatar animation.
### 5.2 Practical AU–Blendshape Mappings in Toolkits
Aside from research prototypes, several practical toolkits embed AU–blendshape mappings:

- The Eurographics “Facial Action Units to Avatars” work targets ARKit-compatible blendshape sets, highlighting the relevance of AU-to-ARKit mappings shared in practitioner resources.[^1][^11]
- Microsoft’s HeadBox toolkit defines 48 FACS-based blendshapes as part of its standard avatar library, ensuring that each avatar can express a consistent set of AUs.[^10]
- Community projects for Unity and other engines implement FACS-based facial expression animation pipelines, in which AU intensities are mapped to engine-specific blendshape parameters.[^14]

These efforts collectively illustrate a maturing ecosystem where AU-labeled motion capture or recognition outputs are used to drive blendshape rigs in mainstream engines and asset libraries.
## 6. Microexpression Analysis in the AU Space
### 6.1 Survey Work on Automatic Microexpression Recognition
Several survey papers consolidate the state of automatic microexpression recognition (MER).
Zhang and Arandjelović (2021) review MER over the previous decade, defining microexpressions as small, rapid facial movements lasting roughly 0.05–0.2 seconds and often revealing genuine emotional states.[^5]
They categorize work by feature types, learning methods, datasets, and outstanding challenges, noting that microexpressions are difficult to recognize due to their brief duration and low intensity.[^5]

Xie et al. (2020) present another overview of microexpression analysis, explicitly highlighting three directions for MER: macro-to-micro adaptation, key apex-frame-based recognition, and AU-based recognition.[^6]
This review stresses the importance of AU-based representations for microexpressions and discusses synthetic data generation and microexpression spotting as auxiliary tasks to improve recognition performance.[^6]

More recent overviews continue this dual psychological–computer-vision perspective, summarizing four main tasks in microexpression analysis: spotting, recognition, AU detection, and generation; they underline that ME analysis requires modeling both fine spatial changes and rapid temporal dynamics.[^7]
### 6.2 AU-Based Microexpression Recognition Frameworks
Specific frameworks leverage AUs directly for microexpression recognition.
A 2025 Scientific Reports paper proposes an action-unit-based microexpression recognition framework, motivated by the idea that microexpressions correspond to subtle temporal patterns of AUs rather than solely to global appearance changes.[^15]
The method uses AU sequences extracted from video to classify microexpressions, illustrating that AU time-series can capture the essential dynamics despite the very short expression length.[^15]

Another line of work focuses on emotion-specific AUs for microexpression recognition, re-analyzing microexpression datasets (e.g., CASME II, SAMM, CAS(ME)2) to identify which AUs and regions of interest (RoIs) are most informative for particular emotion classes.[^8]
This analysis yields new AU-based RoIs that reportedly improve recognition accuracy by a few percentage points compared with baseline regions, underscoring the utility of AU-driven feature design for microexpressions.[^8]

In addition, some works formulate microexpression spotting and recognition directly in FACS terms, using FACS-coded microexpression events as ground truth and treating AU activation onsets and offsets as key cues for spotting.[^6][^7]
## 7. AU-Driven Animation of Micro and Macro Expressions
### 7.1 AU-Driven Micro/Macro Animation
![](https://upload.wikimedia.org/wikipedia/commons/3/38/1106_Side_Views_of_the_Muscles_of_Facial_Expressions.jpg)
Labeled anatomical illustration of facial expression muscles in the head and neck.
Recent work explicitly connects AU-driven blending with animation of both macro and micro facial expressions.
A 2024 paper on “Micro and macro facial expressions by driven animations in realistic facial avatars” (title from arXiv summary) investigates how to drive realistic avatars with AU-based controls to reproduce micro and macro expressions.[^2]
Although details are still emerging, the paper belongs to a growing set of work treating AUs as a common language between psychological descriptions of expressions and avatar animation systems.

The AU-Blendshape framework also includes AU-driven control interfaces where users can mix AUs and tune activation strength to generate fine-grained expressions, including subtle variants.[^12]
While not exclusively focused on microexpressions, these interfaces are capable of producing brief, low‑amplitude AU activations characteristic of microexpressions.
### 7.2 AU Time-Series as Control Signals for Blendshapes
The Eurographics “Facial Action Units to Avatars” work demonstrates a complete pipeline: video → AU intensities (via OpenFace) → recurrent mapping → blendshape weights → real-time avatar motion.[^1][^2]
Because the GRU model retains temporal context, rapid AU changes—such as those associated with microexpressions—can, in principle, be translated into equally rapid fluctuations in blendshape weights, provided the AU detector captures them.[^1]

Survey work on microexpression analysis notes AU-based MER approaches as a promising direction, suggesting that the same AU time-series used for recognition could serve as control signals for animation.[^6][^7]
This conceptual bridge allows a unified representation where AU sequences are used for both analysis (recognition, spotting) and synthesis (avatar control) of microexpressions.
## 8. Connecting the Three Layers: Conceptual Synthesis
Across the reviewed literature, a consistent conceptual layering emerges:

1. **Psychological layer (FACS/AUs & microexpressions)**: FACS defines the basic muscle actions (AUs) and how they combine temporally to produce expressions, with microexpressions characterized as short, often low-amplitude AU patterns.[^4][^5][^6]
2. **Intermediate representation (AU time-series)**: Many microexpression analysis methods represent video sequences as time-series of AU intensities, enabling AU-based MER and AU-centric RoI design.[^6][^8][^15]
3. **Animation/geometry layer (blendshapes)**: Blendshape rigs—especially those designed as AU-specific or FACS-inspired—implement geometric deformations corresponding to each AU, so that AU intensities can be mapped to blendshape weights.[^1][^2][^3][^9][^10]

Research on AUBlendSet/AUBlendNet and on AU-to-avatar mapping explicitly occupies the interface between layers 2 and 3, learning robust mappings between AU parameters and blendshape deformations across many identities and styles.[^3][^12][^1]
In parallel, microexpression research on AU-based recognition and AU-derived RoIs occupies the interface between layers 1 and 2, clarifying which AU patterns are discriminative for specific microexpressions.[^8][^15][^6]

Taken together, these strands indicate that microexpressions can be understood as brief, subtle trajectories in AU space and, when rigs are AU-aligned, as brief, subtle temporal envelopes of AU-specific blendshape weights.
## 9. Gaps and Open Research Directions
The surveyed work also highlights several gaps and opportunities at the intersection of blendshapes, AUs, and microexpressions.
### 9.1 Data Limitations and Temporal Resolution
Microexpression datasets are relatively small, often captured in constrained laboratory setups, and their frame rates and annotation granularity vary.[^5][^6]
This constrains the ability of AU detectors—and hence AU-to-blendshape pipelines—to capture very brief AU activations reliably, potentially limiting the fidelity of microexpression animation.

Further, AU-blendshape datasets like AUBlendSet focus mainly on static AU poses and stylized expression manipulation, not explicitly on microexpression-like temporal patterns.[^3]
There is a need for datasets that explicitly couple high-frame-rate AU sequences, microexpression ground truth, and AU-aligned blendshape deformations across identities.
### 9.2 Intensity Calibration and Subtlety
Microexpressions are defined not only by brevity but also by subtlety; their amplitude may be lower than that of deliberate expressions.[^5][^7]
While AU-to-blendshape mappings demonstrate that AU intensities can be mapped to blendshape weights, there is little work on perceptually validated intensity calibration for microexpressions—how small should blendshape weights be, and what temporal profiles are needed, for an avatar’s microexpression to be perceived as authentic rather than exaggerated or missed.

Perceptual studies combining AUBlendNet-style interfaces with human evaluation of micro and macro expressions could clarify these questions.
### 9.3 Cross-Domain Transfer: From Recognition to Animation
Most MER work focuses on recognition accuracy, not on using trained models to drive avatar animation.
The AU-based MER literature provides detailed knowledge about which AU patterns (and regions) are most informative for recognizing microexpressions.[^8][^15]
However, systematic methods for taking those AU patterns and using them to design or modulate AU–blendshape animation (for example, constructing emotion-specific AU envelopes for avatars) are still rare.

Some recent efforts on microexpression generation and data augmentation indicate interest in synthesizing microexpressions, but these are usually framed as data generation tasks for training recognizers rather than as animation design for real-time avatars.[^6][^7]
Bridging this gap could involve reusing MER-trained AU models as controllers or priors for microexpression animation.
### 9.4 Style, Identity, and Generalization
AUBlendSet and AUBlendNet explicitly tackle style generalization by learning AU-specific basis vectors across many identities and character styles.[^3][^12]
However, microexpression perception may be sensitive to individual differences in facial morphology and motion, raising questions about how well AU-to-blendshape models generalize across identities for subtle expressions.

Exploring whether AU–blendshape mappings trained on macro expressions transfer reliably to microexpressions, and whether additional style or identity conditioning is needed, remains an open challenge.
## 10. Summary
Current literature establishes a clear conceptual pathway connecting FACS AUs, microexpressions, and blendshape-based facial animation.
Microexpression research defines microexpressions as brief, subtle AU patterns and develops AU-based representations and recognition frameworks, while AU–blendshape datasets and mapping methods show how AUs can be translated into blendshape controls for stylized facial animation.[^1][^3][^5][^6][^7][^15]

Nonetheless, explicitly microexpression-focused AU–blendshape animation remains underexplored: most AU–blendshape work targets general expression manipulation and stylization, and most MER work targets recognition rather than synthesis.
Future research that jointly models AU sequences, microexpression semantics, and AU-aligned blendshape rigs—preferably with perceptual evaluation—has the potential to deliver avatars capable of expressing credible microexpressions driven by psychologically grounded AU control.

---

## References

1. [AU-Blendshape for Fine-grained Stylized 3D Facial Expression ...](https://arxiv.org/html/2507.12001v1) - Facial expressions can be accurately described as combinations of corresponding AUs, and the FACS pr...

2. [[PDF] Blendshape-augmented Facial Action Units Detection - RPI ECSE](https://sites.ecse.rpi.edu/~cvrl/Cuiz/papers/Cui2020b.pdf) - 3D AU blendshapes provided by the FaceGen1, which provides morphs at three different levels: phoneme...

3. [Facial microexpressions: what they are and what they mean | TSW](https://www.tsw.it/en/journal-eng/research-experiences/facial-microexpressions-what-they-are-and-what-they-tell-us/) - Facial microexpressions are short-lived but provide information on basic emotions. Let's discover th...

4. [[PDF] Facial Action Units to Avatars - Eurographics](https://diglib.eg.org/bitstreams/b0754f8b-5151-42c0-b3d4-89047e031724/download) - Abstract. Action Units and blend shapes are two frameworks to describe facial movement. However, map...

5. [[PDF] Извлечение и анализ микровыражений в задаче нейросетевого ...](https://inf.grid.by/jour/article/download/1370/1139) - В статье для анализа мимики из системы кодирования лицевых движений (Facial Action Coding System,. F...

6. [FACS-based Facial Expression Animation in Unity - GitHub](https://github.com/dccsillag/unity-facs-facial-expression-animation) - In this project, simple facial expression animation was implemented in Unity via a FaceManager.cs sc...

7. [Roznn/facial-blendshapes: Model for predicting perception ... - GitHub](https://github.com/Roznn/facial-blendshapes) - In this paper, we explore the noticeability of blendshapes under different activation levels, and pr...

8. [The Facial Action Coding System for Characterization of Human ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC7264164/) - The aim of this systematic review is to give an overview of how FACS has been used to investigate hu...

9. [ARKit to FACS: Blendshape Cheat Sheet](https://melindaozel.com/arkit-to-facs-cheat-sheet/) - Whether or not you are FACS-savvy, if you want a clearer breakdown of ARKit facial expression shapes...

10. [Facial Action Coding System - Wikipedia](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) - The Facial Action Coding System (FACS) is a system to taxonomize human facial movements by their app...

11. [[PDF] Facial Action Coding System Emily B. Prince, Katherine B. Martin ...](https://local.psy.miami.edu/faculty/dmessinger/c_c/rsrcs/rdgs/emot/FACSChapter_SAGEEncyclopedia.pdf) - However, comprehensive coding is also time consuming; a one second video clip can take fifteen minut...

12. [[PDF] Leveraging Blendshapes for Realtime Physics-Based Facial ...](https://theses.hal.science/tel-02862792v1/file/2017_BARRIELLE_archivage.pdf) - We wish to build a face physical system by leveraging existing blendshapes characters, taking advant...

13. [Micro and macro facial expressions by driven animations in realistic ...](https://arxiv.org/html/2408.16110v1) - Focusing on VHs expressions, the use of blendshapes [3] , applied in games and other applications, a...

14. [Action unit based micro-expression recognition framework for driver ...](https://www.nature.com/articles/s41598-025-12245-7) - Micro-expressions are subtle, fleeting movements using the same muscles, but lasting only 1/25 to 1/...

15. [Blendshape-Based Facial Animation Using OPF and Random Forest](https://ieeexplore.ieee.org/iel8/6287639/6514899/11395971.pdf) - This paper presents a facial animation method that utilizes a supervised classifier model to identif...

