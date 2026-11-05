# Major Project: Deepfake Video Detection

Face-crop based detection with a fine-tuned EfficientNet-B0, evaluated with video-grouped cross-validation on
UADFV and SDFVD 2.0. A Streamlit app demonstrates the predictions.

## Pipeline

1. **Frames:** 32 frames sampled evenly from each video.
2. **Face crop:** MediaPipe Face Mesh (468 landmarks) finds the face; a crop 1.3x the landmark box is taken
   (centre crop if no face is found), resized to 160x160.
3. **Classifier:** ImageNet-pretrained EfficientNet-B0, fine-tuned end to end (random crop, flip, brightness and
   contrast jitter, label smoothing 0.05, AdamW, one-cycle learning rate).
4. **Video decision:** mean of the frame probabilities; threshold 0.5. The app also marks 35-65 % as uncertain.

A feature-based baseline (landmark geometry, frequency spectrum, face-boundary statistics, frozen CNN embeddings;
gradient boosting, logistic regression, Transformer) is included for comparison (`extract.py`, `evaluate.py`).
It reached only about 88 % on UADFV, which is why the pipeline above fine-tunes the CNN.

## Setup (Windows PowerShell, Python 3.9-3.12)

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install "numpy<2" "opencv-python<4.11" mediapipe==0.10.14 torch torchvision scikit-learn joblib matplotlib tqdm streamlit
```

## Run

```powershell
# 1. face crops (about 3 min for UADFV, about 6 min for SDFVD 2.0; cached, safe to re-run)
python crops.py --data uadfv="C:\path\to\UADFV" sdfvd="C:\path\to\SDFVD2.0"

# 2. honest evaluation: pooled 5-fold CV, videos grouped by source
#    UADFV: a real video and the fake made from it share a number (0000.mp4 / 0000_fake.mp4)
#    SDFVD: real_v10_aug_3 / fake_vs10_aug_2 share source 10
python finetune_cv.py --dataset uadfv sdfvd --epochs 4 --fpv 4 --step 2 --group-regex "^(\d+)|vs?(\d+)_aug" --out results_ft_pooled_grouped2

# UADFV alone (pairs grouped)
python finetune_cv.py --dataset uadfv --epochs 6 --group-regex "^(\d+)" --out results_ft_grouped

# 3. cross-dataset test (train on one, test on the other)
python finetune_cross.py --train uadfv --test sdfvd --epochs 6 --fpv 8 --step 2
python finetune_cross.py --train sdfvd --test uadfv --epochs 4 --fpv 4 --step 2

# 4. final model on all videos
python finetune_final.py --dataset uadfv sdfvd --epochs 4 --fpv 4 --step 2

# 5. predict from the command line, or launch the demo app
python predict_ft.py "C:\path\to\video.mp4"
streamlit run app.py
```

## Optional accuracy options (measured: no benefit)

`--nets N` trains N networks (per fold, or for the final model) and averages them; `--tta` averages each frame with
its mirror image at test time. On UADFV (pairs grouped, 5-fold) `--nets 3 --tta` gave 97.96 % accuracy (precision
100 %, recall 95.92 %, AUC 1.0), the same as a single network within noise, at three times the training time. The
default single network is therefore recommended.

## Robustness, augmentation and Grad-CAM

* **Self-test (no PyTorch or data needed):** `python selftest.py` checks the degradations, the Grad-CAM maths and the
  grouping patterns.
* **Robustness test:** accuracy of held-out videos when their face crops are blurred, shrunk, JPEG-compressed or noisy
  (13 conditions), from a full source-grouped cross-validation:

  ```powershell
  python robustness_cv.py --dataset uadfv sdfvd --group-regex "^(\d+)|vs?(\d+)_aug" --aug none --out rob_none
  python robustness_cv.py --dataset uadfv sdfvd --group-regex "^(\d+)|vs?(\d+)_aug" --aug strong --out rob_strong --compare rob_none\robustness.json
  ```
  Degradations are applied to the saved 160 px crops, which only approximates real video compression.
* **Training augmentation:** `--aug strong` randomly blurs, shrinks, JPEG-compresses or adds noise to training crops
  (`finetune_cv.py`, `finetune_cross.py`, `finetune_final.py`, `robustness_cv.py`).
* **Grad-CAM:** the app shows heatmaps of the face regions that push a frame towards FAKE (sidebar checkbox). It shows
  where the model looked, not proof that a region was manipulated.
* **Two models in the app:** put several `.pt` files in `models/` and pick one in the sidebar. To make the robust one:
  `python finetune_final.py --dataset uadfv sdfvd --epochs 4 --fpv 4 --step 2 --aug strong --out models/finetune_robust.pt`

### Measured robustness (pooled UADFV + SDFVD 2.0, 1025 videos, source-grouped 5-fold, video accuracy)

| Condition | No augmentation | `--aug strong` |
|---|---|---|
| clean | 94.3 % | 90.1 % |
| blur sigma 1 / 2 / 3 | 68.0 / 57.6 / 56.5 % | 88.4 / 84.5 / 79.4 % |
| downscale 0.5 / 0.35 / 0.25 | 66.4 / 57.8 / 54.2 % | 87.8 / 86.2 / 81.2 % |
| JPEG quality 70 / 40 / 20 | 90.4 / 81.0 / 73.0 % | 90.8 / 89.9 / 88.5 % |
| noise sigma 5 / 10 / 20 | 89.0 / 50.9 / 48.9 % | 90.0 / 90.2 / 81.6 % |
| mean of the 12 degraded conditions | 66.1 % | 86.5 % |
| worst condition | 48.9 % (chance) | 79.4 % |

Findings: the standard model is fragile (mild blur, shrinking or noise drops it towards chance, with real videos
flagged as fake); strong augmentation removes most of that at a cost of about 4 points on clean video.
It did NOT improve cross-dataset generalisation (train UADFV, test SDFVD: 57.1 % accuracy, AUC 0.66, versus 58.0 %,
AUC 0.68 without it). The reference paper reports no robustness or cross-dataset test on UADFV.

## Measured results

All numbers come from videos the model never saw in training.

| Test | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| UADFV, pairs grouped, 5-fold (seed 42 run, no grouping) | 97.96 % | 97.96 % | 97.96 % | 0.997 |
| UADFV, pairs grouped, 5-fold (seed 7) | 98.98 % | 100 % | 97.96 % | 1.000 |
| UADFV, pairs grouped, 5-fold, `--nets 3 --tta` | 97.96 % | 100 % | 95.92 % | 1.000 |
| UADFV inside the pooled, source-grouped run | 97.96 % | 97.96 % | 97.96 % | 0.998 |
| SDFVD 2.0 inside the pooled, source-grouped run | 95.25 % | 91.46 % | 100 % | 0.996 |
| All 1025 videos, pooled, source-grouped | 95.51 % | 92.02 % | 99.81 % | 0.995 |
| Train UADFV, test SDFVD | 58.0 % | 92.7 % | 18.9 % | 0.68 |
| Train SDFVD, test UADFV | 51.0 % | 55.6 % | 10.2 % | 0.69 |

Frame-level accuracy of the pooled model is about 93 % (UADFV about 96 %).

## Comparison with the reference paper (UADFV)

The reference paper (Scientific Reports 2026, FastMTCNN + Dlib-68 + VGG16 + LSTM + Transformer + DDM) reports
99.6 % accuracy on UADFV.

* Its headline accuracy is higher than the results above (about 98-99 %).
* Its split is one 80/20 split of 98 videos, about 20 test videos, so 99.6 % cannot be a per-video figure;
  the paper does not say how it was scored. Here every video is scored by a model that did not see it, with each
  real/fake pair, and each SDFVD source, kept in one fold.
* The paper reports no cross-dataset test on UADFV; here it is reported and is weak (51-58 %).
* This project additionally evaluates SDFVD 2.0 and ships a working prediction app.

## Limitations

* Accuracy is high within the training datasets and drops to 51-58 % across datasets: the model does not generalise
  to other sources, qualities or forgery methods.
* On SDFVD 2.0 it leans towards "fake" (44 of 456 real videos flagged).
* UADFV has only 98 videos, so one video moves accuracy by about 1 percentage point; results vary by about
  1 point between runs.
* Only UADFV and SDFVD 2.0 were used; FF++, DFDC and Celeb-DF were not available.
* Perfect prediction is not achievable; the app reports uncertainty instead of forcing a verdict.

## Files

| File | Purpose |
|---|---|
| `crops.py` | save face crops per video |
| `finetune_cv.py` | fine-tuned model, grouped cross-validation (one or several datasets) |
| `finetune_cross.py` | cross-dataset test |
| `finetune_final.py` | train the final model on all videos |
| `predict_ft.py` | command-line prediction |
| `app.py` | Streamlit demo |
| `extract.py`, `evaluate.py`, `train_final.py`, `predict.py` | feature-based baseline |
| `dfd/` | shared code (data loading, frame sampling, features, baseline models) |
| `robustness_cv.py` | accuracy under blur / downscale / JPEG / noise |
| `degrade.py`, `gradcam.py` | degradations and Grad-CAM helpers |
| `selftest.py` | checks that need no PyTorch or data |
| `make_synthetic.py` | writes synthetic features to smoke-test the baseline code |