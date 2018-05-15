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
usage: multi_ae.py [-h] [-d DATA_DIR] [-o OUTPUT_DIR] [-m MODEL_DIR] [-b N]
                   [--init-dim INIT_DIM] [--log-interval N] [--epochs N]
                   [--lr LR] [--no-cuda] [--seed S]

PyTorch FACESWAP Example

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        input data directory
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output data directory
  -m MODEL_DIR, --model-dir MODEL_DIR
                        model pth directory
  -b N, --batch-size N  input batch size for training (default: 64)
  --init-dim INIT_DIM   the number of initial channel (default: 32)
  --log-interval N      how many batches to wait before logging training
                        status
  --epochs N            number of epochs to train (default: 100000000)
  --lr LR               learning rate (default: 5e-5)
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
```

## How to

### 1. 얼굴 데이터를 준비한다.

- 데이터를 준비해둘 경로의 기본값은 ./data이다.
- 이 때, 사람 별로 얼굴 데이터를 폴더에 담아 준비해야 한다.
    - 예: ./data/trump, ./data/cage, ...
- 얼굴 이미지는 bounding box에 맞게 오려져 mean facial landmark에 맞추어 transformation 되어 있는 상태여야 한다.

### 2. 모델을 학습한다.

```
$ python multi_ae.py -d ./data --init-dim 32 -m ./model/multi_32 -o ./output/multi_32 --log-interval 1
```

- 학습된 모델은 기본으로 `./model/multi/`에 저장된다. 위의 명령처럼 직접 지정해줄 수도 있다.
- 모델 입력/출력/타겟 이미지는 기본으로 `./output/multi/`에 저장된다. 위의 명령처럼 직접 지정해줄 수도 있다.
- `log_interval`마다 인당 epoch 100씩 공용 encoder와 사람별 decoder를 학습한다.
    - `log_interval` 내에서 학습을 마치면 모델을 저장하고, 모델 입력/출력/타겟 이미지를 저장한다.
    
- 모델 구조 (`init_dim=32, code_dim=1024`)
```
- encoder
input:  (  3,  64,  64)
conv1:  ( init_dim,  64,  64) = ( 32,  64,  64)
conv2:  ( 64,  64,  64)
conv3:  (128,  64,  64)
conv4:  (256,  64,  64)
reshape:  (256*64*64, )
linear1:  (code_dim, ) = (1024, )
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
