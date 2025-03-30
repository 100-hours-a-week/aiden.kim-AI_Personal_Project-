# Almond Varieties Classification using CNN, VGG16, and InceptionNet

본 프로젝트는 Kaggle의 아몬드 이미지 데이터셋을 기반으로, 터키 지역의 네 가지 아몬드 품종(`Ak`, `Nurlu`, `Kapadokya`, `Sıra`)을 분류하는 딥러닝 모델을 개발하고 성능을 비교한 연구입니다.

<br/>

## 프로젝트 개요

- **목표**: 다양한 CNN 기반 모델을 활용하여 아몬드 품종 분류 성능 비교
- **모델 비교**:
  - Pre-trained 모델: VGG16, InceptionV3
  - Custom CNN 모델: 3개의 Conv + MaxPooling 계층 + Dense 레이어 구성
- **주요 실험**: Dense 레이어 노드 수(256, 512)에 따른 성능 변화 관찰
- **이론적 기반**: 보편적 근사 정리 (Universal Approximation Theorem)

<br/>

## 데이터셋

- **출처**: Kaggle Almond Varieties
- **총 이미지 수**: 1,556장
- **클래스 수**: 4종 (Ak, Nurlu, Kapadokya, Sıra)
- **데이터 분할 비율**:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

<br/>

## 사용한 모델

### VGG16
- **특징**: 3x3 커널, 13 Conv + 3 Dense 구조
- **장점**: 단순하고 강력한 구조, 전이 학습 가능
- **단점**: 파라미터 수가 많고 연산량이 큼

### InceptionV3
- **특징**: 다양한 크기의 필터를 병렬로 적용하는 Inception Module 사용
- **장점**: 효율적인 연산, 고성능
- **단점**: 구조가 복잡함

### Custom CNN
- **구성**: 3 Conv + 3 MaxPooling + Dense(256/512) + Dropout(0.3)
- **장점**: 간결한 구조, 빠른 학습 속도, 자원 효율성
- **단점**: 복잡한 특징 추출에는 한계

<br/>

## 실험 결과

| Model            | Dense 노드 수 | Test Accuracy |
|------------------|---------------|---------------|
| VGG16            | 256           | 0.95          |
| VGG16            | 512           | 0.95          |
| InceptionV3      | 256           | 0.91          |
| InceptionV3      | 512           | 0.96          |
| Custom CNN       | 256           | 0.97          |
| Custom CNN       | 512           | 0.97          |

- **Custom CNN 모델이 가장 높은 정확도(0.97)를 달성**
- Dense 레이어 노드 수를 512로 확장하면 더 복잡한 특징 학습에 유리

<br/>

## 이론적 배경: 보편적 근사 정리

> 단일 은닉층을 가진 신경망도 충분한 수의 뉴런과 적절한 활성화 함수가 있으면 어떤 연속 함수든 근사 가능하다.

$$
f(x) = \sum_{i=1}^{N} \alpha_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)
$$

- $x$: 입력 벡터
- $\sigma$: ReLU 등 비선형 활성화 함수
- $\alpha_i$: 출력 가중치
- $\mathbf{w}_i$, $b_i$: 은닉층 가중치, 편향
- $N$: 은닉 노드 개수

<br/>

## 구현 요약

- **프레임워크**: TensorFlow / Keras
- **데이터 분할**: `splitfolders`
- **Image Augmentation**: 회전, 이동, 확대 등 적용
- **Optimizer**: Adam
- **Callbacks**: EarlyStopping, ModelCheckpoint
- **시각화**: loss/accuracy 그래프 출력

```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
