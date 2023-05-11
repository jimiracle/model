# 🛸 Diff-SVC Colab Training Tutorial

## Data

### Training
* local에서 돌릴때와 동일한 조건의 음성 데이터가 필요
* 단 슬라이싱이 저절로 되지 않아 10~15초 단위로 슬라이싱한 파일을 .zip 형태로 업로드

### Inference
* 출력을 원하는 노래의 반주가 없는 음성 데이터
* 44100Hz & 16비트
* 슬라이싱 x, 통째로 넣음


## GitHub에서 바로 Colab Notebook 열기
`Batch_Diff_SVC_Inference.ipynb` 파일을 Colab에서 바로 열고 싶다면,  
https://github.com/yaiconwithminsu/model/blob/main/diff-svc-colab/Batch_Diff_SVC_Inference.ipynb 에서  
https://colab.research.google.com/yaiconwithminsu/model/blob/main/diff-svc-colab/Batch_Diff_SVC_Inference.ipynb 로,  
`https://github.com` 을 `github`로 변경하고, 앞에 `https://colab.research.google.com/` 넣어주면 됨.

## 1. 처음부터 학습을 시작한다면

* Diff_SVC_training_notebook.ipynb
Step 0,1,2-A,3,4,4-A,5
순서대로 진행 후, 원하는만큼 학습시킨 후 Batch_Diff_SVC_Inference.ipynb로 출력물을 낼 수 있음

## 2. 학습을 재개하고 싶다면

Required : 전 학습으로 부터 얻어진 .ckpt file/config file이 필요

처음 학습할 때 Use custom save directory라는 옵션을 통해서 구글 드라이브에 저장해놓으면 가능

* Diff_SVC_training_notebook.ipynb **Step 0,1,2-B,3,4,(4-A),5**
순서대로 진행 후, 원하는만큼 학습시킨 후 Batch_Diff_SVC_Inference.ipynb로 출력물을 낼 수 있음
* 4-A의 경우 데이터 전처리 단계인데, binarzied된 data가 존재한다면 skip가능
* (colab의 경우 40~60분의 데이터 전처리가 오래걸려서 처음 4-A를 실행할때 save_binary_to_gd를 체크하는 것을 추천)

<br>
