# faceswap-pytorch

## Dependencies
this code is tested on python>=3.5 and CUDA 9.1 environment with additional packages below:
```
torch==0.5.0
torchvision==0.2.1
opencv-python==3.3.0.10
numpy==1.14.2
matplotlib==2.2.2
```

## usage statement

```
usage: train.py [-h] [-d DATA_DIR]
                [-m {AE64,AE128}]
                [-o OUTPUT_DIR] [-b BATCH_SIZE] [--epochs EPOCHS]
                [--sub-epoch SUB_EPOCH] [--lr LR] [--no-cuda] [--seed SEED]
                [--fix-enc] [--mask-loss]

PyTorch FACESWAP Example

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        input data directory
  -m {AE64,AE128}
                        select a model to train
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output dir name (which will become output dir name)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 64)
  --epochs EPOCHS       number of epochs to train (default: 100)
  --sub-epoch SUB_EPOCH
                        number of loop in an epoch for each face (default:
                        100)
  --lr LR               learning rate (default: 5e-5)
  --no-cuda             disables CUDA training
  --seed SEED           random seed (default: 1)
  --fix-enc             fix encoder a
```

## How to

### 1. 얼굴 데이터를 준비한다.

- 데이터를 준비해둘 경로의 기본값은 ./data이다.
- 이 때, 사람 별로 얼굴 데이터를 폴더에 담아 준비해야 한다.
    - 예: ./data/trump, ./data/cage, ...
- 얼굴 이미지는 bounding box에 맞게 오려져 있어야 한다.

### 2. 모델을 학습한다.

```
$ python train.py -d ./data -m AE64 -o AE64_SEGMENT
```

- 학습된 모델은 `-n` 인자에 해당하는 폴더에 저장된다. 위의 경우, `./output/AE64_SEGMENT`에 저장된다.
- 학습 중에 일정한 간격으로 모델 입력/출력/타겟 이미지를 저장한다. 위의 경우, `./output/AE64_SEGMENT/face_id`에 저장된다
- `log_interval`마다 인당 epoch `inner_loop`씩 공용 encoder와 사람별 decoder를 학습한다.
    - `log_interval` 내에서 학습을 마치면 모델을 저장하고, 모델 입력/출력/타겟 이미지를 저장한다.

- AE64 모델 구조 (`init_dim=128, code_dim=1024`의 경우)
```
- encoder
input:  (  3,  64,  64)
conv1:  ( init_dim=128,  32,  32)
conv2:  ( 256,  16, 16)
conv3:  (512,  8, 8)
conv4:  (1024,  4,  4)
reshape:  (1024*4*4, )
linear1:  (code_dim=1024, )
linear2:  (1024*4*4, )
reshape:  (1024, 4, 4)
upscale:  ( 512, 8, 8)

- decoder
input:    (512,   8,   8)
upscale1: (256,  16,  16)
upscale2: (128,  64,  32)
upscale3: ( 64,  64,  64)
conv&sigmoid: (3, 64, 64)
```