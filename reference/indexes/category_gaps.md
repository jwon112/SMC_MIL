# WSI literature coverage

Generated from indexes; rerun `python reference/tools/coverage.py`.

Registered papers: 43; reviewed atoms: 48; pending candidates: 8.

Counts describe this curated library, not completeness of the published literature.
A reviewed atom supports its bounded statement; a relevant method paper is not direct heart-transplant clinical validation.

## Organ

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| heart | 15 | 16 |
| kidney | 2 | 3 |
| other | 14 | 16 |
| general | 15 | 17 |

## Analysis units

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| patch | 15 | 18 |
| region | 2 | 2 |
| slide | 19 | 22 |
| core | 1 | 1 |
| specimen | 10 | 10 |
| event | 2 | 2 |
| patient | 12 | 12 |
| not_applicable | 11 | 13 |

## Stains / modality

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| HE | 22 | 25 |
| IHC | 4 | 4 |
| special | 5 | 6 |
| molecular | 1 | 1 |
| not_applicable | 19 | 21 |

## Methods

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| mil | 10 | 12 |
| self_supervised | 7 | 9 |
| transformer | 4 | 4 |
| hierarchical | 2 | 2 |
| multiscale_fusion | 1 | 2 |
| cross_stain_alignment | 1 | 1 |
| graph_fusion | 1 | 1 |
| feature_distillation | 1 | 1 |
| handcrafted | 4 | 4 |
| stain_augmentation | 1 | 1 |
| stain_normalization | 1 | 1 |
| calibration | 4 | 4 |
| audit | 14 | 14 |
| guideline | 4 | 6 |
| spatial_transcriptomics | 1 | 1 |
| missingness_modeling | 1 | 1 |
| transfer_learning | 3 | 4 |

## Validation design

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| benchmark | 20 | 21 |
| retrospective | 14 | 15 |
| external_cohort | 5 | 6 |
| site_separated | 1 | 1 |
| prospective | 1 | 1 |
| not_applicable | 5 | 7 |

## Research focus

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| cardiac | 15 | 16 |
| mil | 7 | 9 |
| encoder | 7 | 9 |
| multiscale | 2 | 3 |
| multi_slide | 2 | 2 |
| stain_fusion | 2 | 2 |
| shortcut | 4 | 5 |
| domain_shift | 5 | 6 |
| evaluation | 16 | 19 |
| calibration | 5 | 6 |
| label_validity | 13 | 13 |
| interpretability | 3 | 3 |

## Endpoints

| Facet | Papers | Reviewed atoms |
|---|---:|---:|
| cardiac_rejection | 10 | 11 |
| clinical_severity | 3 | 3 |
| tumor_classification | 10 | 11 |
| histology_scoring | 1 | 1 |
| survival | 6 | 6 |
| representation_learning | 6 | 7 |
| methodology | 13 | 15 |
| tissue_segmentation | 1 | 2 |
| pathogen_detection | 1 | 1 |
| myocardial_injury | 1 | 1 |
| myocarditis | 1 | 1 |
| fibrosis_quantification | 1 | 1 |
| cardiomyopathy_classification | 1 | 1 |

## Topic coverage

| Topic | Routed papers | Reviewed routed atoms |
|---|---:|---:|
| cardiac_allograft_pathology | 12 | 13 |
| computational_pathology_mil | 7 | 7 |
| foundation_encoders | 7 | 8 |
| multiscale_wsi_representation | 5 | 6 |
| stain_and_domain_shift | 5 | 6 |
| weak_and_longitudinal_labels | 5 | 5 |
| patient_grouped_evaluation | 5 | 7 |
| external_validation | 7 | 7 |
| calibration_and_ensembles | 3 | 5 |
| interpretability | 4 | 3 |
| event_stain_fusion | 3 | 3 |
| acquisition_shortcuts | 3 | 3 |
| biopsy_label_validity | 1 | 1 |
| presence_mask_and_missingness | 6 | 6 |
| imbalance_handling | 2 | 2 |
| rare_positive_evaluation | 2 | 2 |
| threshold_selection | 1 | 1 |
| small_sample_validation | 3 | 3 |
| autopsy_biopsy_transfer | 11 | 12 |

## Registered paper comparison

| Paper | Year | Organ | Units | Methods | Validation |
|---|---:|---|---|---|---|
| [agniel2018_healthcare_process](../cards/agniel2018_healthcare_process.json) |  | general | patient | audit | retrospective |
| [amancherla2025_spatial_transcriptomics](../cards/amancherla2025_spatial_transcriptomics.json) |  | heart | core | spatial_transcriptomics | retrospective |
| [carriero2025_imbalance_ml](../cards/carriero2025_imbalance_ml.json) |  | general | not_applicable | calibration | benchmark |
| [cawley2010_selection_bias](../cards/cawley2010_selection_bias.json) |  | general | not_applicable | audit | benchmark |
| [che2018_grud](../cards/che2018_grud.json) |  | general | patient | missingness_modeling | benchmark |
| [chen2024_uni](../cards/chen2024_uni.json) |  | other | patch, slide | self_supervised, transformer | benchmark |
| [chicco2020_mcc](../cards/chicco2020_mcc.json) |  | general | not_applicable | audit | benchmark |
| [collins2024_tripod_ai](../cards/collins2024_tripod_ai.json) |  | general | not_applicable | guideline | not_applicable |
| [colvin2015_aha_amr](../cards/colvin2015_aha_amr.json) |  | heart | slide, event | guideline | not_applicable |
| [goorbergh2022_imbalance](../cards/goorbergh2022_imbalance.json) |  | general | not_applicable | calibration | benchmark |
| [ilse2018_abmil](../cards/ilse2018_abmil.json) |  | general | patch, slide | mil | benchmark |
| [lipkova2022_crane](../cards/lipkova2022_crane.json) |  | heart | patch, slide | mil | external_cohort |
| [lipton2014_f1_threshold](../cards/lipton2014_f1_threshold.json) |  | general | not_applicable | calibration | benchmark |
| [lu2021_clam](../cards/lu2021_clam.json) |  | other | patch, slide | mil | external_cohort, benchmark |
| [moons2025_probast_ai](../cards/moons2025_probast_ai.json) |  | general | not_applicable | guideline | not_applicable |
| [peyster2021_cache_grader](../cards/peyster2021_cache_grader.json) |  | heart | slide | handcrafted | external_cohort |
| [riley2020_sample_size](../cards/riley2020_sample_size.json) |  | general | not_applicable | audit | not_applicable |
| [saito2015_precision_recall](../cards/saito2015_precision_recall.json) |  | general | not_applicable | audit | benchmark |
| [seraphin2023_cardiac_ssl](../cards/seraphin2023_cardiac_ssl.json) |  | heart | patch, slide | mil, self_supervised | external_cohort |
| [shao2021_transmil](../cards/shao2021_transmil.json) |  | other | patch, slide | mil, transformer | benchmark |
| [stewart2005_ishlt_acr](../cards/stewart2005_ishlt_acr.json) |  | heart | slide | guideline | not_applicable |
| [tellez2019_stain](../cards/tellez2019_stain.json) |  | other | patch | stain_augmentation, stain_normalization | benchmark |
| [varoquaux2018_small_sample_cv](../cards/varoquaux2018_small_sample_cv.json) |  | general | not_applicable | audit | benchmark |
| [zerbe1988_rejection_sampling](../cards/zerbe1988_rejection_sampling.json) | 1988 | heart | specimen | audit | retrospective |
| [hauck1989_myocarditis_sampling](../cards/hauck1989_myocarditis_sampling.json) | 1989 | heart | specimen, patient | audit | retrospective |
| [nakhleh1992_biopsy_autopsy](../cards/nakhleh1992_biopsy_autopsy.json) | 1992 | heart | specimen, patient | audit | retrospective |
| [guo2017_calibration](../cards/guo2017_calibration.json) | 2017 | general | not_applicable | calibration | benchmark |
| [husain2017_c4d_autopsy](../cards/husain2017_c4d_autopsy.json) | 2017 | heart | specimen, patient | audit | prospective, retrospective |
| [howard2021_site_signatures](../cards/howard2021_site_signatures.json) | 2021 | other | slide, patient | audit | site_separated |
| [li2021_dsmil](../cards/li2021_dsmil.json) | 2021 | other | patch, slide | mil, self_supervised, multiscale_fusion | benchmark |
| [chen2022_hipt](../cards/chen2022_hipt.json) | 2022 | other | patch, region, slide | self_supervised, transformer, hierarchical | benchmark |
| [dwivedi2022_multistain_graph](../cards/dwivedi2022_multistain_graph.json) | 2022 | other | slide, patient | graph_fusion, mil | benchmark |
| [lutnick2022_histocloud](../cards/lutnick2022_histocloud.json) | 2022 | kidney, other | patch, slide | transfer_learning | retrospective |
| [mcdonald2022_biopsy_autopsy](../cards/mcdonald2022_biopsy_autopsy.json) | 2022 | heart | specimen, patient | audit | retrospective |
| [zaizen2022_autopsy_biopsy_ai](../cards/zaizen2022_autopsy_biopsy_ai.json) | 2022 | other | patch, specimen, patient | transfer_learning | retrospective |
| [zhang2022_dtfd](../cards/zhang2022_dtfd.json) | 2022 | other | patch, slide | mil, hierarchical, feature_distillation | benchmark |
| [amemiya2024_biopsy_wholeheart](../cards/amemiya2024_biopsy_wholeheart.json) | 2024 | heart | specimen, patient | handcrafted, audit | retrospective |
| [jaume2024_madeleine](../cards/jaume2024_madeleine.json) | 2024 | kidney, other | patch, slide | self_supervised, mil, cross_stain_alignment | benchmark |
| [peyster2024_clinical_trajectory](../cards/peyster2024_clinical_trajectory.json) | 2024 | heart | slide, event | handcrafted | retrospective |
| [vorontsov2024_virchow](../cards/vorontsov2024_virchow.json) | 2024 | other | patch, slide, specimen | self_supervised, transformer, mil | benchmark |
| [burk2025_transplant_autopsy](../cards/burk2025_transplant_autopsy.json) | 2025 | heart | specimen, patient | audit | retrospective |
| [mori2025_cardiomyopathy_small_sample](../cards/mori2025_cardiomyopathy_small_sample.json) | 2025 | heart | region, patient | handcrafted | retrospective |
| [shen2025_songci](../cards/shen2025_songci.json) | 2025 | general, other | patch, slide, specimen | self_supervised, transfer_learning | retrospective, external_cohort |

## Open research questions

- **heart-event-fusion**: Does multi-slide/multi-stain event aggregation improve cardiac rejection beyond slide-to-event averaging? (Method analogues reviewed; direct cardiac comparative evidence still needed)
- **heart-scale-fusion**: Does physical-MPP or coordinate-linked scale fusion improve ACR/AMR under patient-grouped external validation? (General method evidence only)
- **source-shortcuts**: How much performance remains after Exp3/MRXS, scanner, stain, and slide-count controls? (Local audit question; literature cannot establish the local effect)
- **rare-positives**: What precision and calibration can be established with few independent positive patients? (Selected-page methodological evidence reviewed 2026-10-07: imbalance, thresholds, MCC and small-sample validation; direct cardiac rare-positive calibration evidence remains absent from this index)
- **clinical-utility**: Does routine cardiac WSI improve clinically meaningful decisions in prospective settings? (Prospective task-matched validation not yet indexed)
- **encoder-transfer**: Which encoder transfers to cardiac H&E and non-H&E under identical splits? (Noncardiac encoder evidence reviewed; cardiac comparison pending)
- **heart-autopsy-biopsy-transfer**: Does pathology-confirmed autopsy tissue improve rejection detection on independent biopsy patients, compared with biopsy-only training on identical splits? (Cardiac sampling/paired pathology and noncardiac AI precedents reviewed, including SongCi reported clinical-pathology transfer (supplement details unaudited). Direct matched cardiac rejection AI improvement remains unestablished by indexed sources. SMC autopsy dataset availability reported; inventory and labels not audited.)

## Review queue

| Priority | Candidate | Question / next action |
|---|---|---|
| 1 | [A machine learning algorithm improves the diagnostic accuracy of the histologic component of antibody mediated rejection (AMR-H) in cardiac transplant endomyocardial biopsies.](https://pubmed.ncbi.nlm.nih.gov/38677634/) | AMR-H region classification versus complete pAMR event diagnosis. / Acquire full text; verify patient split and annotation-level leakage. |
| 1 | [An integrated clinical-histopathologic prediction model for cardiac allograft rejection: Translating machine learning into clinical risk frameworks.](https://pubmed.ncbi.nlm.nih.gov/42070726/) | EHR+morphology incremental value and prospective feature availability. / Local PubMed print exists; acquire full text and verify time windows/splits. |
| 1 | [Removal of Autopsy Artifacts from Whole Slide Images Using Unpaired Adversarial Training](https://www.nature.com/articles/s41379-020-0462-y.pdf) | Does unpaired autopsy-to-biopsy appearance translation preserve diagnostically relevant lesions and improve downstream biopsy tasks? / Acquire official USCAP 2020 abstract 19 pages and inspect results/figures; locate any full follow-up publication. Source PDF retrieval failed in current environment; do not treat search snippet as reviewed evidence. |
| 1 | [Computational pathology assessments of cardiac stromal remodeling: Clinical correlates and prognostic implications in heart transplantation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935495/) | Repeated-biopsy morphology and prognosis; distinguish CAV/death from rejection. / Acquire source; inspect endpoint timing, cohort reuse and validation. |
| 1 | [Cross-Stain Contrastive Learning for Paired Immunohistochemistry and Histopathology Slide Representation Learning](https://arxiv.org/abs/2512.03577) | Is pairing/alignment required, and how are missing stains handled? / Review camera-ready source, pairing rules and fusion ablations. |
| 2 | [A multimodal whole-slide foundation model for pathology](https://www.nature.com/articles/s41591-025-03982-3) | Slide encoder versus current task-trained MIL. / Review source, train/test overlap and slide/report availability. |
| 2 | [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) | Does image-text pretraining help cardiac/non-H&E representations? / Acquire source and review slide-level benchmark scope. |
| 2 | [Transformer-based unsupervised contrastive learning for histopathological image classification](https://www.sciencedirect.com/science/article/pii/S1361841522002043) | Encoder transfer and training-domain overlap. / Acquire lawful author manuscript; inspect methods and benchmark splits. |
