# Continual Learning-Based Website Fingerprinting on Tor

본 프로젝트는 Deep Fingerprinting(DFNet)에 Continual Learning (특히 EWC: Elastic Weight Consolidation)을 적용하여 Catastrophic Forgetting을 완화하는 실험을 다룹니다.

## 🧩 주요 기능
- Incremental task 기반 DFNet 학습
- EWC 손실항 추가 가능 (`--lamb`)
- Task별 accuracy/log 기록

---

## ⚠️ NVIDIA GPU 및 CUDA 사용 시 주의사항

- 이 프로젝트는 GPU 가속을 위해 `tensorflow-gpu==2.10.0`을 사용합니다.
- GPU 사용을 위해 아래 시스템 구성 필요:

| 구성 요소 | 권장 버전 |
|-----------|-----------|
| CUDA Toolkit | 11.2 |
| cuDNN | 8.1 |
| NVIDIA GPU 드라이버 | 최신 |

- CUDA 및 cuDNN 설치 방법: [TensorFlow 공식 가이드](https://www.tensorflow.org/install/gpu)


## 🛠️ 가상환경 설정 방법 (Python venv 기준)

아래 명령어를 사용해 프로젝트와 동일한 Python 환경을 구성할 수 있습니다.

```bash
# 1. 가상환경 생성 (예: nAIvis)
python -m venv nAIvis

# 2. 가상환경 활성화
# Windows
nAIvis\Scripts\activate
# macOS/Linux
source nAIvis/bin/activate

# 3. 필요한 패키지 설치
pip install -r requirements.txt
```


## 🚀 실행 예시

```bash
python main.py --first_task 69 --inc_task 25 --first_epochs 70 --inc_epochs 20 --lamb 10000
```
