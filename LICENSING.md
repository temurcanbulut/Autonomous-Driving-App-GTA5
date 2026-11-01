# Licensing scope

The MIT license in LICENSE applies only to Temur Can Bulut's original source code and documentation text. It does not grant rights to GTA V, third-party scene imagery, source datasets, pretrained model parameters, or the separately released trained checkpoints.

The recorded replay is a research diagnostic: it renders previously recorded results, makes no model call, and sends no game input. Its scene illustrations and model results are not evidence of end-to-end autonomous driving.

## Data, model and image boundaries

- The documented source dataset is Playing for Data by Richter, Vineet, Roth and Koltun (ECCV 2016), described by its source as research/educational material: https://download.visinf.tu-darmstadt.de/data/from_games/ . Preserve the citation and applicable source terms. Raw dataset images are not supplied as a dataset here.
- Checkpoints, third-party pretrained parameters and dataset-derived illustrations are excluded from this source-code MIT grant. Consult the original terms and MODEL_CARD.md before redistributing or using those artifacts. No new commercial-use permission for those artifacts is asserted here.
- Installed Python packages retain their own licenses. Do not interpret the GitHub license badge as relicensing third-party content.


## Bundled model configuration

`Autonomous-Driving-App-GTA5/yolov3.cfg` is a third-party YOLO v3 model configuration, outside this repository's original-code MIT grant. The upstream [Darknet configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) and [upstream license](https://github.com/pjreddie/darknet/blob/master/LICENSE) are maintained by Joseph Redmon and contributors; their terms are not replaced by this LICENSE. Weights and other pretrained parameters remain separate artifacts. No model configuration or weight file is changed by this licensing documentation.
