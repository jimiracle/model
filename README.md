# 🎵 노래민수야고마워
> Diff-SVC 모델를 사용해 노래의 음성을 원하는 음성로 변환하는 프로젝트

# 👀 Preview

### 주호민 버전 재즈
<audio src="preview/joohomin_jazz_input.mp3" controls preload="metadata"></audio>

### 침착맨 버전 재즈 (Diff-SVC)
<audio src="preview/chim_jazz_output.wav" controls preload="metadata"></audio>

침착맨 말하기 데이터 15분, 노래 부르기 데이터 15분으로  
Diff-SVC 모델을 학습해, 주호민 버전 재즈를 침착맨 버전 재즈로 변환한 결과임.

# 🛸 Diff-SVC Colab Training Tutorial
[**바로가기**](diff-svc-colab/README.md)


# 🚀 Diff-SVC Local Training Tutorial

## 1. 레포지토리 fork하기

해당 레포지토리에서는, diff-svc-local 폴더를 fork하면 됨.  

처음부터 시작하고 싶다면, 
- [**Official - prophesier/diif-svc**](https://github.com/prophesier/diff-svc)
- [**Updated - UtaUtaUtau/diff-svc**](https://github.com/UtaUtaUtau/diff-svc)

Official 레포지토리보다는 UtaUtaUtau 레포지토리를 fork하기를 매우 추천함.  
(pitch를 더 보정해 깔끔한 오디오 파일이 출력되도록 업데이트되어 있기 때문임.)

UtaUtaUtau 레포지토리 코드를 사용한다면, [pyworld](https://pypi.org/project/pyworld/) 패키지를 0.3.1 버전으로 설치하길 바람.  
(fork한 후, requirements.txt에 pyworld==0.3.1 적길 바람. 소스 레포에서는, requirements_short.txt에 버전 없이 pyworld라고 적혀 있음.)

## 2. conda 환경 구축하기 (Setting up the Environment)

anaconda3와 cuda 설치 안되어있다면, 설치하기 

`python=3.8` 인 conda 환경 구축하기 (3.8 버전 혹은 3.9 버전 설치 바람.)

```bash
conda create -n diff-svc python=3.8

# conda 환경 이름은 개인 선택 (예시에서는 diff-svc로 진행함.)
```

환경 활성화하기. (모델 돌리기 전에, 항상 먼저 실행해줘야 함.)

```bash
conda activate diff-svc 
```

## 3. requirements 설치하기 (Install requirements)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

pytorch, torchvision 버전에 맞춰 설치하기.

```bash
pip install -r requirements.txt
pip install -r requirements_short.txt
```

requirements.txt를 통해 필요한 패키지 설치하기.

## 4. pretrained model checkpoints 다운받기

**해당 구글드라이브 폴더에서 다운 받으면 됨.** : [checkpoints - Google Drive](https://drive.google.com/drive/folders/1Zk-gOalI4FtDs63G9twAM7HUImQnn14O)

checkpoints 폴더 생성해서, 다운 받은 파일 넣어주면 됨.

`nsf_hifigan_xxxx.zip` https://github.com/openvpi/vocoders/releases 여기서도 다운 받을 수 있음.  
(44.1kHz 모델 훈련하고 싶으면, nsf_hifigan(44.1kHz vocoder) 파일 꼭 다운받아야함.)

`nsf_hifigan_finetune`, `nsf_hifigan_onnx` 필요 없음.

[Nehito-Diff-SVC_44khz.zip 다운받기](https://github.com/MLo7Ghinsan/MLo7_Diff-SVC_models/releases/tag/models) 로 들어가,   
Nehito Pretrained Model checkpoint 파일 다운받고, 압축을 풀어 nehito_44khz라는 폴더를 만들어 넣어주기.

<img src="https://github.com/yaiconwithminsu/model/assets/88659167/ee714a75-de64-4bde-a376-2129d1c2af3a"> 

위와 같이, `checkpoints` 폴더 구조를 이루면 됨.
(여기서필요한 것은 0102_xiaoma_pe, hubert, nehito_44khz, nsf_hifigan 임.  
 *chim, minus 폴더는 개인 프로젝트 ckpt 파일을 저장하기 위한 폴더로 생성한 것임.)

<img src="https://user-images.githubusercontent.com/88659167/236933872-d1016d2f-3fed-470f-95eb-2f712a714e4f.png" width="550" height="350">

## 5. 데이터 준비하기 (Dataset Preparation)

타깃 대상의 보컬 데이터 준비하기

- 배경음악 없어야 함.
- 단독 목소리이어야 함.
- 말하기 / 노래 데이터 사용 가능함.
*일반적으로, 노래 데이터가 더 믾이 사용되는 편임.
- `.wav` , `.ovv` 파일 형태이어야 함.
- 5~15초 길이의 오디오 파일이어야 함. 
(더 길어도 가능하지만, CUDA OOM에 문제가 생길 수 있음.)
- 샘플링 레이트는 24kHz 이상이 좋은 편임. 16kHZ 이하면 안됨.
- 폴더명, 파일명에 공백이 있으면 안됨.
- 파일명은 고유해야함. (다른 폴더에 있어도, 파일명이 같으면 안됨.)
- 최소 6개의 파일 이상 있어야함.

+) 노래 부르는 영상에서, 반주 제외하고, 음성 부분만 추출해줌.

- **GAUDIO STUDIO** : [https://studio.gaudiolab.io/](https://studio.gaudiolab.io/)

+) 44100Hz & 16비트 변환

- **wav to mp3** : [https://js-audio-converter.com/kr/](https://js-audio-converter.com/kr/)

### 데이터 정제

### 1. de-noise & de-reverb

데이터에 노이즈가 많다면, 정제해주는 것이 좋음.

- **Adobe Podcast :** [https://podcast.adobe.com/enhance](https://podcast.adobe.com/enhance)

### 2. loudness normalization

- **Adobe Audition :** [https://helpx.adobe.com/audition/using/match-loudness.html](https://helpx.adobe.com/audition/using/match-loudness.html)

### 3. audio slicing

굳이 따로 슬라이싱할 필요 없음. 프로그램이 알아서 슬라이싱해줌.

다만, 사이트가 더 빠르게 슬라이싱한다고 생각 들어, 사이트를 통해 따로 슬라이싱한 후, 데이터 폴더에 넣는 것을 추천함.

따로 슬라이싱 원하다면,

15초 간격으로 음성 분리해서 파일로 저장해줌.

- **Free Batch Music Splitter** : [https://www.fosshub.com/Free-Batch-Music-Splitter.html](https://www.fosshub.com/Free-Batch-Music-Splitter.html)

<img src="https://user-images.githubusercontent.com/88659167/236933877-39a2468c-d074-40f3-9adb-58e4fc255d78.png" width="300" height="220">

## 6. 전처리 (Preprocessing)

### 슬라이싱

`preprocess` 라는 폴더 생성하고, 학습 데이터 넣어주기.

그 후,

```bash
python sep_wav.py
# 슬라이싱하는 데, 시간이 오래 걸릴 수 있음
```

완료된 후, `preprocess_out` 폴더에 슬라이싱된 데이터 확인 가능함.

해당 데이터를 `data/raw/{speaker_name}` 를 생성해 넣어주기.

(여기서, 이미 따로 오디오 슬라이싱 작업을 진행한 경우 해당 단계를 패스함.)

### 이진화

원시 오디오 데이터를 이진 파일로 변환하기.

`config.yaml` (24kHz vocoder 사용하는 경우)

`config_nsf.yaml` (44.1kHz vocoder 사용하는 경우)

confg 파일에서 {speaker_name}으로 변경하기 (경로 마지막 부분 nsf라고 써있는 부분을 변경하기)

config 파일 수정하기 전, 복사본 생성해두기.

```bash
binary_data_dir: data/binary/{speaker_name}
# 전처리 데이터 파일 경로

raw_data_dir: data/raw/{speaker_name}
# 원시 데이터 파일 경로

speaker_id: {speaker_name}
# 타깃 대상 이름

work_dir: checkpoints/{speaker_name}

load_ckpt: checkpoints/nehito_44khz/nehito_ckpt_steps_1000000.ckpt
# 더 좋은 ouput을 위해, pretrained model의 ckpt 파일 불러오기 
# nehito 같은 경우, 남자 목소리를 학습한 것으로, 남자 목소리에 더 적합함.
# 여자 목소리는 liee 사용하길 바람. (ckpt 파일 찾아 다운로드 받길 바람.)
```

<img src="https://user-images.githubusercontent.com/88659167/236933881-eacc86f1-7f43-4907-8c5c-50490625d68b.png" >

(선택사항) pitch extraction 알고리즘 변경

CREPE 알고리즘 (pitch extraction)

```bash
use_crepe: true
# 더 좋은 결과를 위해 true로 설정함. 빠른 전처리를 원하면 false로 설정하는 것이 일반적이지만,
# 개인적으로 false로 설정하고, 빠른 학습을 지향하는 것이 좋은 것 같음. 더 좋은 결과 나오는지 잘 확인 안되는 것 같음.
```

### 환경설정 (꼭 하기)

```bash
# Windows (cmd)
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0 # GPU 사용 설정
python preprocessing/binarize.py --config training/config_nsf.yaml

# Linux
export PYTHONPATH=. 
CUDA_VISIBLE_DEVICES=0 
python preprocessing/binarize.py --config training/config_nsf.yaml
```
Windows 환경인지, Linux 환경인지 꼭 확인하고, 그에 맞는 명령어를 실행해야 함.

<img src="https://user-images.githubusercontent.com/88659167/236933883-1617c728-cde8-4977-8e30-9629f9510e82.png" width="600" height="400">

이때,

```bash
File "preprocessing/binarize.py", line 6, in <module>
    from utils.hparams import set_hparams, hparams
ModuleNotFoundError: No module named 'utils.hparams'
```

오류가 뜨면,

최상위 폴더로 `[binarize.py](http://binarize.py)` 를 옮기고,

```bash
python binarize.py --config training/config_nsf.yaml
```

이 오류가 나는 것은 환경변수 설정이 잘 안되어서임.

Windows 환경 / Linux 환경에 맞는 환경변수 명령어 (`set PYTHONPATH=.` / `export PYTHONPATH=.`) 를 실행했는지 확인하길 바람.

그래도 안되면 위와 같이 하길 바람.

<br>
그리고 아래와 같은 오류가 뜨면,

```bash
RuntimeError: stft requires the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release.
```

<img src="https://user-images.githubusercontent.com/88659167/236933885-a50b0726-45cb-4176-8db4-99cb59ad0b8f.png">

`modules/nsf_hifigan/nvSTFT.py` 에서 해당 코드에 `return_complex=False` 파라미터 추가하기.

## 7. 훈련 (Training)

### config 파일 수정

```bash
max_sentences: 88 # VRAM의 80G에 해당하므로, VRAM에 따라 조정해줘야 함.
# CUDA Out of Memory 에러가 뜨면,
# config 파일에서, max_sentences (batch_size)를 줄여주면 됨.


# 선택사항
endless_ds: false
# true로 설정하면, 1,000개의 epoch를 하나의 epoch로 취급함.

val_check_interval: 2000
# test dataset에 대한 추론할 때, 2,000 step마다 checkpoint를 저장함.

num_ckpt_keep: 10
# checkpoint 파일이 최신 10개로 유지됨.
```

### 훈련 실행

```bash
# Windows
set CUDA_VISIBLE_DEVICES=0 # GPU 사용 설정
python run.py --config training/config_nsf.yaml --exp_name {project_name} --reset

# Linux
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config_nsf.yaml --exp_name {project_name} --reset
```

checkpoints는 `{your_project_folder}/checkpoints/{youer_project_name}` 에서 `val_check_interval` step (default는 2,000 step)마다 저장됨.

+) Tensorboard (선택사항)

tensorboard를 통해, 진행사항을 보고 싶다면,

```bash
tensorboard --logdir=checkpoints/{your_project_name}/lightning_logs/lastest
```

실행 후, [http://localhost:6006/](http://localhost:6006/) 에서 Tensorboard 확인하기.

<img src="https://user-images.githubusercontent.com/88659167/236933869-b785a31c-6bd8-41fd-bc09-a3a15836dc32.png" width=500 height=300>

## 8. 추론 (Inference)

input 데이터 음성을 타깃 대상의 음성으로 변환하는 단계임.
ex) 뉴진스 노래를 침착맨 목소리로 변경하기

이를 “랜더링 (rendering)”이라고 부르는 편임.

추론 단계에서, `[infer.py](http://infer.py)` 혹은 `inference.ipynb` 파일를 사용하게 됨.

사용의 편의를 위해 `inference.ipynb` 파일 사용을 추천함.

infery.py를 돌리고 싶다면, `python infer.py` 실행하면 됨.

Jupyter Notebook을 사용하기 위해,

```bash
conda install notebook
conda install ipykernel
ipython kernel install --user --name=diff-svc # kernel 이름은 각자 편의대로.
```

- `project_name` : training에 사용된 프로젝트 이름; config 파일에서 설정한 이름
- `model_path` : checkpoint 파일 경로
- `config_path` : config 파일 경로
- `wav_fn` : input 오디오 파일 경로
- `wav_gen` : output 오디오 파일이 저장될 경로
**.wav 대신 .flac와 같은 다양한 오디오 파일 형식을 지원함.*
- `key=0` (선택사항) : 피치 조정
ex) 남자 목소리를 여자 목소리로 변경한다면, 8 혹은 12로 설정
- `pndm_speedup` (선택사항) : 추론 가속 계수
default값이 20은, 1000/20=50 diffusion step이 추론 중에 실행된다는 의미.
더 빠른 추론을 원하면, 해당 값을 높이면 됨.
- `use_crepe` (선택사항) : true로 설정하면, CREPE 알고리을 사용해 추론 도중에 pitch extraction 한다는 의미. false로 설정하면, 더 빠르게 추론 결과를 얻을 수 있음.

output 파일은 `wav_gen` 에서 설정한 경로로 저장됨.

<br>

## Referece

- [Diff-SVC local guidebook](https://docs.google.com/document/d/1XQlOcv1Xx2BpVeSUyDleb2Th9sYOnG1miFrGZ8uGcsE/edit)
- [인공 르르땅으로 핫한 Diff-svc 학습방법(로컬 환경, 집 컴퓨터) : Youtube](https://www.youtube.com/watch?v=8hJ1Wullg_g)
- [인공 르르땅으로 핫한 Diff-svc 학습방법(로컬 환경, 집 컴퓨터) : GitHub](https://github.com/wlsdml1114/diff-svc)
- [The Beginner’s Guide to Diff-SVC](https://diff-svc.gitbook.io/the-beginners-guide-to-diff-svc/) 
