## Flags
- S: Started
- M: Half
- C: Close to completition
- D: Done

- W: Worker
- P: Added onto pipeline
- D: Data Processed (Finished)

## TODO
- Download Dataset Splits
- Separate into scenes with pyscenedetect [WC]
- Calculate Optical Flow via OpenCV Farneback
- Calculate Quality Score via VQA (Named TQA: Technical Quality Assesment) [WC]
- Save all metrics onto SQL
- Filter as desired

## Dataset Splits
- 75K Shutterstock (CleanVid)
- 225K Youtube (HDVILA)
Total 300K

## Changelog
- 2/02/2024 Init repo

## Sources

https://github.com/sayakpaul/single-video-curation-svd
https://arxiv.org/abs/2311.15127

### SVD's techniques

- Captioning: Too poor imo, not going to be added
- Clip Extraction: Added
- Optical Flow: To be Added
- Similarity Aesthetics: Replaced by DOVER Aesthetic Scoring
- Text Detection: In Consideration