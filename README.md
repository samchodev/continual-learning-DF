# Continual Learning-Based Website Fingerprinting on Tor

본 프로젝트는 Tor 네트워크의 익명성을 위협하는 웹사이트 핑거프린팅(Website Fingerprinting, WF) 공격에 대응하기 위해, Deep Fingerprinting (DFNet) 모델에 평생학습(Continual Learning) 기법 중 하나인 Elastic Weight Consolidation (EWC)를 적용하여 치명적 망각(Catastrophic Forgetting) 문제를 완화하는 방법을 제안한다.

---

## 📌 연구 배경

웹사이트 핑거프린팅(WF) 공격은 암호화된 트래픽의 메타데이터만으로 사용자가 방문한 웹사이트를 식별할 수 있는 심각한 보안 위협이다. Tor 네트워크는 노드 릴레이와 고정 크기 셀을 통해 익명성을 보호하지만 WF 공격은 이를 우회할 수 있다.

딥러닝 기반 DFNet은 기존 WF 모델보다 높은 식별 정확도를 보였지만 고정된 환경에서만 작동하여 변화하는 웹 환경에 대응하기 어렵다는 한계가 있다.

이에 따라, 본 연구는 **Continual Learning** 기법 중 **EWC**를 DFNet에 적용하여 새롭게 등장하는 웹사이트 트래픽을 학습하면서도 기존 지식의 성능 저하를 완화하는 모델을 제안한다.

---

## 📌 주요 기능
- Incremental task 기반 DFNet 학습
- EWC 손실항 추가 가능 (`--lamb`)
- Task별 accuracy/log 기록

---

## 📌 데이터셋 구성

본 프로젝트에 사용된 데이터셋은 Mathews et al.의 SoK 논문에서 제안된 "Monitored Set"을 기반으로 수집된 웹사이트 트래픽을 활용하였으며, 해당 데이터의 구축 과정 및 특성은 다음 논문에 자세히 기술되어 있다:

> N. Mathews et al., “SoK: A Critical Evaluation of Efficient Website Fingerprinting Defenses,” *In Proceedings of the 2023 IEEE Symposium on Security and Privacy (SP)*, pp. 969–986, May 2023.

본 repository에 업로드한 데이터셋은 해당 원본 데이터를 기반으로 가공한 형태로, 각 `Direction_Sequence`는 길이 10,000으로 전처리된 상태이다.  
이 데이터는 실험 재현성을 고려하여 정규화 및 포맷 통일 과정을 거친 결과물이다.

---

## 📌 NVIDIA GPU 및 CUDA 사용 시 주의사항

- 이 프로젝트는 GPU 가속을 위해 `tensorflow-gpu==2.10.0`을 사용합니다.
- GPU 사용을 위해 아래 시스템 구성 필요:

| 구성 요소 | 권장 버전 |
|-----------|-----------|
| CUDA Toolkit | 11.2 |
| cuDNN | 8.1 |
| NVIDIA GPU 드라이버 | 최신 |

- CUDA 및 cuDNN 설치 방법: [TensorFlow 공식 가이드](https://www.tensorflow.org/install/gpu)


## 📌 가상환경 설정 방법 (Python venv 기준)

아래 명령어를 사용해 프로젝트와 동일한 Python 환경을 구성할 수 있습니다.

```bash
conda create -n nAIvis python=3.7.7
conda activate nAIvis
pip install -r requirements.txt
```

```bash
# Python 3.7.7 기준
python -m venv nAIvis
nAIvis\Scripts\activate           # Windows
source nAIvis/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

## 📌 GitHub 저장소 클론
```bash
git clone https://github.com/hineugene/continual-learning-DF.git
cd continual-learning-DF
```

## 📌 실행 예시

```bash
python main.py --first_task 69 --inc_task 25 --first_epochs 70 --inc_epochs 20 --lamb 10000
```

---

## 📌 실험 결과

### ✅ Nonbase-line vs EWC vs Joint-line

| 설정   | T1 정확도 | T2 정확도 | T2에서의 T1 정확도 | 평균 정확도 |
|--------|-----------|-----------|--------------------|--------------|
| None   | 0.9861    | 0.9815    | (0.044)            | 0.9838       |
| EWC    | 0.9186    | 0.7610    | (0.409)            | 0.8398       |
| Joint  | 0.9623    | 0.9885    | (0.927)            | 0.9754       |

>  **EWC 적용 시**, 기존 Task 성능이 크게 저하되지 않으며, 기존 지식을 효과적으로 보존함을 확인하였습니다.


### ✅ 파라미터 조정 실험 결과

1️⃣ λ(lambda) 값에 따른 성능 변화

| Lambda (λ) | T1 정확도 | T2 정확도 | T2에서의 T1 정확도 |
|------------|-----------|-----------|--------------------|
| 1          | 0.9258    | 0.9500    | (0.018)            |
| 5          | 0.9219    | 0.9430    | (0.173)            |
| 100        | 0.9214    | 0.8740    | (0.632)            |
| 1000       | 0.9226    | 0.8033    | (0.679)            |

> λ가 커질수록 기존 Task(T1)의 정확도는 유지되지만, 새로운 Task(T2) 학습 성능은 하락함 → **하이퍼파라미터 튜닝 필요**


2️⃣ 클래스 비율 (Task1 : Task2)에 따른 성능 변화

| 클래스 비율 | T1 정확도 | T2 정확도 | T2에서의 T1 정확도 |
|-------------|-----------|-----------|--------------------|
| 90:5        | 0.9317    | 0.9400    | (0.637)            |
| 70:25       | 0.9146    | 0.7657    | (0.464)            |
| 50:45       | 0.9185    | 0.6882    | (0.594)            |

> 초기 Task에 더 많은 클래스를 포함할수록 기존 지식 보존 효과가 커짐


3️⃣ 에폭 비율 (Task1 : Task2)에 따른 성능 변화

| 에폭 비율  | T1 정확도 | T2 정확도 | T2에서의 T1 정확도 |
|------------|-----------|-----------|--------------------|
| 20:20      | 0.8343    | 0.7717    | (0.346)            |
| 50:20      | 0.9072    | 0.8257    | (0.534)            |
| 100:20     | 0.9305    | 0.7850    | (0.699)            |

> Task 1에 더 많은 학습 비중(epoch)을 할당하면 **망각 현상 완화**, T2 성능도 안정적으로 유지됨


## 📌 결과 요약

- EWC 기반 Continual Learning은 기존 WF 모델이 겪는 **Catastrophic Forgetting 문제를 효과적으로 완화**함
- Task 2 학습 이후에도 기존 웹사이트(Task 1)의 식별 정확도를 **평균 40% 이상 유지**
- λ(lambda), 클래스 비율, 에폭 비율 등 **하이퍼파라미터 조정이 성능 유지에 중요한 역할**을 함
- 모든 데이터를 한 번에 학습하는 Joint 방식보다 현실성은 낮지만, **실제 적용 가능한 전략으로서 유의미한 성능을 확보**

---

## 📌 향후 연구 방향

- 다양한 Continual Learning 기법(i.e. SI, MAS, Replay 등)과의 비교 실험을 통해 **최적의 전략 탐색**
- Task 수 확장 및 오픈월드 설정 등 **현실에 가까운 시나리오로 일반화 성능 평가**
- WF 모델과 Tor 네트워크 방어 기법 간의 **대응 전략 연구** (적대적 학습, 방어 모델 통합 등)
- 실시간 트래픽 분류나 온라인 러닝 구조로 확장하여 **운영 가능한 프라이버시 공격 시스템 구현**

즉, 본 연구는 변화하는 환경 속에서도 안정적으로 동작할 수 있는 웹사이트 핑거프린팅 모델의 가능성을 제시하며, 향후 Tor 기반 보안/공격 연구에 기초 자료로 활용될 수 있다.
