# ISIC 2024 - Skin Cancer Detection with 3D-TBP - Images & Tabular Data

## Acknowledgment
These notebooks published by other Kaggle members really helped me getting started:
- https://www.kaggle.com/code/nadhirhasan/tabular-based-pytorch-ann
- https://www.kaggle.com/code/lorrainmorlet/cnn-with-val-overfitting

## Description
Skin cancer can be deadly if not caught early, but many populations lack specialized dermatologic care. Over the past several years, dermoscopy-based AI algorithms have been shown to benefit clinicians in diagnosing melanoma, basal cell, and squamous cell carcinoma. However, determining which individuals should see a clinician in the first place has great potential impact. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

Dermatoscope images reveal morphologic features not visible to the naked eye, but these images are typically only captured in dermatology clinics. Algorithms that benefit people in primary care or non-clinical settings must be adept to evaluating lower quality images. This competition leverages 3D TBP to present a novel dataset of every single lesion from thousands of patients across three continents with images resembling cell phone photos.

This competition challenges you to develop AI algorithms that differentiate histologically-confirmed malignant skin lesions from benign lesions on a patient. Your work will help to improve early diagnosis and disease prognosis by extending the benefits of automated skin cancer detection to a broader population and settings.

## Dataset Description
The dataset consists of diagnostically labelled images (SLICE-3D dataset) with additional metadata. The images are JPEGs. The associated .csv file contains a binary diagnostic label (target), potential input variables (e.g. age_approx, sex, anatom_site_general, etc.), and additional attributes (e.g. image source and precise diagnosis).

To mimic non-dermoscopic images, this competition uses standardized cropped lesion-images of lesions from 3D Total Body Photography (TBP). Vectra WB360, a 3D TBP product from Canfield Scientific, captures the complete visible cutaneous surface area in one macro-quality resolution tomographic image. An AI-based software then identifies individual lesions on a given 3D capture. This allows for the image capture and identification of all lesions on a patient, which are exported as individual 15x15 mm field-of-view cropped photos. The dataset contains every lesion from a subset of thousands of patients seen between the years 2015 and 2024 across nine institutions and three continents.

The following are examples from the training set. 'Strongly-labelled tiles' are those whose labels were derived through histopathology assessment. 'Weak-labelled tiles' are those who were not biopsied and were considered 'benign' by a doctor.

![Image examples](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4972760%2F169b1f691322233e7b31aabaf6716ff3%2Fex-tiles.png?generation=1717700538524806&alt=media)

### Additional metadata
#### Train file only
- target: Binary class {0: benign, 1: malignant}.
- lesion_id: Unique lesion identifier. Present in lesions that were manually tagged as a lesion of interest.
- iddx_full: Fully classified lesion diagnosis.
- iddx_1: First level lesion diagnosis.
- iddx_2: Second level lesion diagnosis.
- iddx_3: Third level lesion diagnosis.
- iddx_4: Fourth level lesion diagnosis.
- iddx_5: Fifth level lesion diagnosis.
- mel_mitotic_index: Mitotic index of invasive malignant melanomas.
- mel_thick_mm: Thickness in depth of melanoma invasion.
- tbp_lv_dnn_lesion_confidence: Lesion confidence score (0-100 scale).

#### Train and test files
- isic_id: Unique case identifier.
- patient_id: Unique patient identifier.
- age_approx: Approximate age of patient at time of imaging.
- sex: Sex of the person.
- anatom_site_general: Location of the lesion on the patient's body.
- clin_size_long_diam_mm: Maximum diameter of the lesion (mm).
- image_type: Structured field of the ISIC Archive for image type.
- tbp_tile_type: Lighting modality of the 3D TBP source image.
- tbp_lv_A: A inside lesion.
- tbp_lv_Aex: A outside lesion.
- tbp_lv_B: B inside lesion.
- tbp_lv_Bext: B outside lesion.
- tbp_lv_C: Chroma inside lesion.
- tbp_lv_Cext: Chroma outside lesion.
- tbp_lv_H: Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).
- tbp_lv_Hext: Hue outside lesion.
- tbp_lv_L: L inside lesion.
- tbp_lv_Lext: L outside lesion.
- tbp_lv_areaMM2: Area of lesion (mm^2).
- tbp_lv_area_perim_ratio: Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.
- tbp_lv_color_std_mean: Color irregularity, calculated as the variance of colors within the lesion's boundary.
- tbp_lv_deltaA: Average A contrast (inside vs. outside lesion).
- tbp_lv_deltaB: Average B contrast (inside vs. outside lesion).
- tbp_lv_deltaL: Average L contrast (inside vs. outside lesion).
- tbp_lv_deltaLBnorm: Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.
- tbp_lv_eccentricity: Eccentricity.
- tbp_lv_location: Classification of anatomical location, divides arms & legs to upper & lower; torso into thirds.
- tbp_lv_location_simple: Classification of anatomical location, simple.
- tbp_lv_minorAxisMM: Smallest lesion diameter (mm).
- tbp_lv_nevi_confidence: Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.
- tbp_lv_norm_border: Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.
- tbp_lv_norm_color: Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.
- tbp_lv_perimeterMM: Perimeter of lesion (mm).
- tbp_lv_radial_color_std_max: Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.
- tbp_lv_stdL: Standard deviation of L inside lesion.
- tbp_lv_stdLExt: Standard deviation of L outside lesion.
- tbp_lv_symm_2axis: Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.
- tbp_lv_symm_2axis_angle: Lesion border asymmetry angle.
- tbp_lv_x: X-coordinate of the lesion on 3D TBP.
- tbp_lv_y: Y-coordinate of the lesion on 3D TBP.
- tbp_lv_z: Z-coordinate of the lesion on 3D TBP.
- attribution: Image attribution, synonymous with image source.
- copyright_license: Copyright license.
