# 🗣️ Voice Interactive Assembly Guide Robot (AI Co-worker)

<img width="1964" height="1053" alt="image" src="https://github.com/user-attachments/assets/7579f045-6c37-435b-9cbf-6dc2bbfbf870" />


<br>

## 🗂️ 목차

### 1. [Project Overview](#-project-overview)
### 2. [Team & Roles](#-team--roles)
### 3. [System Architecture](#-system-architecture)
### 4. [Tech Stack](#-tech-stack)
### 5. [Key Features & Logic](#-key-features--logic)
### 6. [Performance Analysis](#-performance-analysis)
### 7. [Demo Video](#-demo-video)

<br>

---

## 🔍 Project Overview
**"비숙련 작업자도 전문가처럼. 말하면 알아듣고, 불량은 스스로 걸러내는 AI 협동로봇"**

본 프로젝트는 **LLM(Large Language Model)** 기반의 음성 인식 기술과 **YOLOv11-OBB** 비전 기술을 융합한 지능형 조립 보조 시스템입니다.
작업자가 "불량품 찾아줘", "전부 처리해"와 같이 자연어로 명령하면, 로봇이 의도를 파악하여 흩어진 부품 중 불량품만을 골라내거나 필요한 부품을 집어주는(Pick & Place) 역할을 수행합니다. 이를 통해 비숙련자의 작업 효율을 높이고 교육 시간을 단축하는 것이 목표입니다.

<br>

## 👥 Team & Roles

| Name | Role | Responsibility |
|:---:|:---:|:---|
| **Kim Jung-wook** | **Team Leader** <br> **& Scenario Dev** | - **Scenario Logic Design:** 개별/통합 불량 처리 및 조립 검증 시나리오 설계 (State Machine) <br> - **AI Model Training:** 불량/양품 분류를 위한 Custom Dataset 구축 및 YOLO 모델 학습 <br> - **Documentation:** 프로젝트 산출물 관리 및 기술 문서화 |
| **Lee Kang-yeop** | PM & Integration | - **System Integration:** 전체 ROS2 노드(Voice, Vision, Control) 통합 및 일정 관리 <br> - **Safety Logic:** 안전 알고리즘(충돌 감지, 비상 정지) 구현 및 안전영역 설계|
| **Kim Da-bin** | Vision & Environment | - **YOLO Optimization:** YOLOv11-OBB 하이퍼파라미터 튜닝 및 학습 성능 개선 <br> - **Environment Setup:** 작업대 환경 구성 및 데이터셋 라벨링 |
| **Kang Dong-hyuk** | Hardware Support | - **Robot Setup:** 두산 로봇 및 그리퍼 하드웨어 초기 설정 지원 |

<br>

## 🛠 System Architecture

<img width="1875" height="989" alt="image" src="https://github.com/user-attachments/assets/c8418258-fd1f-404f-9eb7-2c53a3b190a5" />

<img width="1906" height="987" alt="image" src="https://github.com/user-attachments/assets/d050bbb2-3812-458f-afaf-f6b69e585955" />

<img width="1782" height="948" alt="image" src="https://github.com/user-attachments/assets/599a2c30-4331-4ca9-a6de-288af41a4947" />

<img width="1543" height="990" alt="image" src="https://github.com/user-attachments/assets/b8256d8a-ed39-4dab-8e65-3ccc64e8f42b" />

<img width="1878" height="937" alt="image" src="https://github.com/user-attachments/assets/82aabcb7-4774-418f-b615-1ad4c95b3d82" />



시스템은 크게 **사용자 인터페이스(Voice)**, **인지(Vision)**, **제어(Control)** 3가지 핵심 노드로 구성됩니다.

1.  **Smart Manager Node (Brain):** 사용자의 음성 명령을 `STT`로 변환하고, `OpenAI(LLM)` & `LangChain`을 통해 의도를 파악하여 로봇에게 작업을 지시합니다.
2.  **YOLO Detection Node (Eyes):** `RealSense` 카메라로 작업대를 촬영하고, `YOLOv11-OBB`로 객체의 종류와 회전 각도(Angle)를 0.1초 내에 식별합니다.
3.  **Robot Control Node (Action):** `Doosan M0609` 로봇을 제어하여 불량품을 폐기하거나 양품을 정렬합니다.

<br>

## 💻 Tech Stack

| Category | Technology |
| :---: | :--- |
| **AI / LLM** | ![OpenAI](https://img.shields.io/badge/OpenAI-GPT_4o-412991?style=flat-square&logo=openai) ![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?style=flat-square) ![Google](https://img.shields.io/badge/Google-STT_TTS-4285F4?style=flat-square&logo=google) |
| **Vision / DL** | ![YOLOv11](https://img.shields.io/badge/YOLO-v11_OBB-00FFFF?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch) |
| **Middleware** | ![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=flat-square&logo=ros) ![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat-square&logo=ubuntu) |
| **Hardware** | ![Doosan](https://img.shields.io/badge/Doosan-M0609-005EB8?style=flat-square) ![RealSense](https://img.shields.io/badge/Intel-RealSense_D435i-0071C5?style=flat-square&logo=intel) |

<br>

## 🚀 Key Features & Logic

### 1. YOLOv11-OBB (Oriented Bounding Box)
일반적인 사각형(Bounding Box)은 회전된 부품을 잡을 때 그리퍼 각도를 알 수 없는 문제가 있습니다.
본 프로젝트는 최신 **YOLOv11-OBB** 모델을 도입하여 객체의 **회전 각도($\theta$)**까지 정밀하게 추론, 로봇이 부품의 각도에 맞춰 손목을 회전하며 잡을 수 있도록 구현했습니다.

<img width="1876" height="895" alt="image" src="https://github.com/user-attachments/assets/5cc40446-a89c-431b-b46b-17acffefa96d" />

<img width="1925" height="940" alt="image" src="https://github.com/user-attachments/assets/06dcfd46-d5b2-4b5b-9af4-4e96a60c6326" />

<img width="1830" height="983" alt="image" src="https://github.com/user-attachments/assets/1a6b6899-bac0-4753-9c63-69574003c4b4" />

<img width="1843" height="903" alt="image" src="https://github.com/user-attachments/assets/3e54b978-4bfb-4c1e-9df1-5c71cb9a9216" />

<img width="1831" height="885" alt="image" src="https://github.com/user-attachments/assets/2c49ee3d-001c-4bb6-8754-70635986446e" />


### 2. Depth Correction Algorithm (5-Point Spatial Averaging)
저가형 Depth 카메라 특성상 발생하는 **'튀는 값(Noise)'** 문제를 해결하기 위해 자체 보정 알고리즘을 개발했습니다.
* **공간적 평균(Spatial):** 객체 중심점 주변 5개 픽셀의 Depth 값을 샘플링하여 평균값 사용
* **시간적 평균(Temporal):** 5프레임 연속 측정 후 평균을 내어 떨림 현상 제거
 
<img width="1823" height="976" alt="image" src="https://github.com/user-attachments/assets/28b1d00e-93ff-4163-8784-dfaca5a6f859" />


### 3. AI Voice Interaction
단순한 키워드 매칭이 아닌, LLM을 활용하여 작업자의 자연스러운 언어를 이해합니다.
* *"이거 불량품이네, 좀 치워줘"* -> **[명령 인식: 불량품 폐기]** -> **[로봇 동작]**

<br>

## 📊 Performance Analysis


* **YOLO Detection Accuracy:** mAP50-95 기준 **97%** 달성
* **Sort Success Rate:** 불량품 분류 성공률 **99%** (시나리오 기반 검증)
* **Voice Recognition:** 작업 현장 소음 환경에서도 핵심 명령 인식률 **80%** 확보

<br>

## 🎥 Demo Video

https://youtu.be/9G9PHG_XNIQ

https://youtu.be/Hi0sv22zA84

https://youtu.be/z8DrpZIK1nQ

https://youtu.be/tmS-EllnBG4

https://youtu.be/ifsjuv5GlYo
