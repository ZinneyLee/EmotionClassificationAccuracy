## Model: Parallel 2D CNN - Transformer Encoder
- Stop at 82 epochs
  - Validation Accuracy: about 82%
  - Test accuracy: about 89%
- Datasets: ESD (Emotional Speech Dataset)
  - Train 15,000
  - Valid 1,000
  - Test 1,500
 
## Evaluation
  - sh run.sh ---> evaluation.py
  - $TEST='{DATASET_LIST_PATH}'
    - DATASET form: 'item  spk  emotion  txt  wav_fn' ---> only use 'emotion, wav_fn'


## Reference Code
- https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch.git
