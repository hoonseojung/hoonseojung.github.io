---
title: "활성화 함수"

categories:
  - DL
tags:
  - [Deep Learning, activation function]

toc: true
toc_sticky: true

date: 2022-07-14
last_modified_at: 2022-07-14
---

# 활성화 함수란?

딥러닝 네트워크에서 노드에 입력된 값들을 비선형 함수에 통과시킨 후 다음 레이어로 전달하는데, 이 때 사용하는 함수를 **활성화 함수(Activation Function)**라고 한다.  

선형 함수가 아니라 비선형 함수를 사용하는 이유는 딥러닝 모델의 레이어 층을 깊게 가져갈 수 있기 때문이다.  

선형함수인 h(x)=cx를 활성화함수로 사용한 3층 네트워크를 예를 들어 식으로 나타내면 y(x)=h(h(h(x)))가 됩니다. 이는 실은 y(x)=ax에서 $$a=c^3$$인 경우와 똑같은 식이다.  

즉, 은닉층이 없는 네트워크로 표현할 수 있기에 Neural network에서 층을 쌓는 이득을 위해서는 활성화 함수(activation function)로 반드시 비선형 함수를 사용해야 한다.  

인공 신경망에서 활성화 함수는 입력 데이터를 다음 레이어로 어떻게 출력하느냐를 결정하는 역할이기 때문에 매우 중요하다.  

즉, 활성화 함수는 입력을 받아서 활성화 또는 비활성화를 결정하는 데에 사용되는 함수이다.  

# 활성화 함수의 종류

## Sigmoid 함수

Sigmoid 함수는 Logistic 함수라고 불리기도 하며, x의 값에 따라 0~1의 값을 출력하는 S자형 함수이다.  

Sigmoid 함수의 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfNDQg/MDAxNTgyNjA2Mzc5MDY5.zxVCMRhPDevOiEqeW9Zld_qIExdLNovxjzMgD5csjKQg.U7vFTEIlTe8blGu1s6wFptli8_X1oe-QOfztJHWN7-og.PNG.handuelly/image.png?type%3Dw800)


Sigmoid 함수 미분 과정
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMjYg/MDAxNTgyNjA2MzkwNTQx.eUt8n14w7VCLfyZxM9zcnaM9WblF9kT7qPfKFnOZajkg.o1XBEQtAaKFbpLBTG_e6XEkV5Vh65HFeQg6OMy_8lQcg.PNG.handuelly/image.png?type%3Dw800)

Sigmoid 함수(좌) & Sigmoid 도함수(우)
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMjI4/MDAxNTgyNjA3MjY0NzIy.tR76IK8YsIL8XORjYDJoMSNBK2nhpUooUhMS6N0d1NUg.bMGce-shmpX6--ck-mvfcTimMjL3UpFw2iSgQgExCR8g.PNG.handuelly/image.png?type%3Dw800)
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTM5/MDAxNTgyNjA3MjcxMjE0.T7uG6k2la4WUJjrRwMHVB3FuidBNK2tYPhXpzSwGDUsg._-ameb8qQWpDkhk5SDUDUU9Bag2GpfN7mw78g_Jc8HQg.PNG.handuelly/image.png?type%3Dw800)

하지만 Sigmoid 함수는 음수 값을 0에 가깝게 표현하기 때문에 입력 값이 최종 레이어에서 미치는 영향이 적어지는 Vanishing Gradient Problem이 발생한다.  

Sigmoid 도함수 그래프에서 미분 계수를 보면 최대값이 0.25이다.  
딥러닝에서 학습을 위해 Back-propagation(역전파)을 계산하는 과정에서 활성화 함수의 미분 값을 곱하는 과정이 포함되는데, Sigmoid 함수의 경우 은닉층의 깊이가 깊으면 오차율을 계산하기 어렵다는 문제가 발상하기 때문에, Vanishing Gradient Problem이 발생한다.  

다시 말해, x의 절대값이 커질수록 Gradient Backpropagation 시 미분 값이 소실될 가능성이 큰 단점이 있다.  

또한, Sigmoid 함수의 중심이 0이 아닌데, 이 때문에 학습이 느려질 수 있는 단점이 있다.  

한 노드에서 모든 파라미터의 미분 값은 모두 같은 부호를 같게 되는데, 같은 방향으로 update되는 과정은 학습을 지그재그 형태로 만드는 원인을 낳는다.  

이러한 문제 때문에 실제로는 잘 사용되지 않는다.  

모든 실수 값을 0보다 크고 1보다 작은 미분 가능한 수로 변환하는 특징을 같이 때문에, Logistic Classification과 같은 분류 문제의 가설과 비용 함수(Cost Function)에 많이 사용된다.  

또한 sigmoid()의 리턴 값이 확률 값이기 때문에 결과를 확률로 해석할 때 유용하다.  

## Tanh 함수

Hyperbolic Tangent Function은 쌍곡선 함수 중 하나로, Sigmoid 함수를 변형해서 얻을 수 있다.  

Tanh 함수의 정의와 미분 과정은 아래 수식과 같은데, $$σ(x)$$는 Sigmoid 함수 식이다.  
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTk0/MDAxNTgyNjA3NzcyOTI3.Oc7diAw06G0vb-b86Wp5O-sy2Oa_bHHZnxnc2ASFYjAg.0dxRIz0WS3xkT51tlC8Yl1w_pDDRWynAaCX4NzxIq3Eg.PNG.handuelly/image.png?type%3Dw800)

Tanh 함수(좌) & Tanh 도함수(우)
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfOSAg/MDAxNTgyNjA3Njg0NDc0.3PxflDp1EDXZVSsOwpBBYUXQP9GLJmPTqz872JMxkc4g.QlUM1zGGT9WK9b7jcLQwhXTAr3Tqp4k2RZuCeidUTOIg.PNG.handuelly/image.png?type%3Dw800)
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTUx/MDAxNTgyNjA3NjkyODMx.YhSLbMr-W9d_FSLnSbnTxdm0GMwYY_Yx6XMTGTI8eWQg.r_CATCsMwW6F4xhrfIFBp3yV6760Ez-sK5phPrUE0jYg.PNG.handuelly/image.png?type%3Dw800)

tanh 함수는 함수의 중심점을 0으로 옮겨 Sigmoid 함수가 갖고 있던 최적화 과정에서 느려지는 문제를 해결했다.  

하지만 미분함수에 대해 일정 값 이상에서 미분 값이 소실되는 Vanishing Gradient Problem은 여전히 남아있다.  

## ReLU 함수

ReLU(Rectified Linear Unit, 경사함수)는 가장 많이 사용되는 활성화 함수 중 하나이다.  

Sigmoid와 tanh가 갖는 Gradient Vanishing 문제를 해결하기 위한 함수이다.  

ReLU 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTA4/MDAxNTgyNjA4MzM2NjI2.BimoIN4e0LyJoEdFhNfXO1q8o9FcMRzCZVnmBNrRqSog.o6c5C2zBc0Wh9YwxR37drT9VvZP_qE4yhSWRw2V68Dkg.PNG.handuelly/image.png?type%3Dw800)

![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfOTIg/MDAxNTgyNjA4MzI2NDA5.e0VyX0yrhE5gtfPjni7IxF5kpArCeByreQsdOMB0240g.CWwTi57bPtAK6C7eLmRn1ED2RE8Lm_C6sVIwMGJS1Akg.PNG.handuelly/image.png?type%3Dw800)
ReLU 함수 그래프

x가 0보다 크면 기울기가 1인 직선, 0보다 작으면 함수 값이 0이 된다. 이는 0보다 작은 값들에서 뉴런이 죽을 수 있는 단점을 야기한다.  

또한 Sigmoid, tanh 함수보다 학습이 빠르고, 연산 비용이 적고, 구현이 매우 간단하다는 특징이 있다.  

## Leaky ReLU

Leaky ReLUU는 ReLU가 갖는 Dying ReLU(뉴런이 죽는 현상)을 해결하기 위해 나온 함수이다.  

함수도 아래와 같이 매우 간단한 형태로 정의된다.  

Leaky ReLU 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMjU5/MDAxNTgyNjA4NzA5MzUx.8bwP5NUnWan-vKq91HKuFL-FdZyG-nLVx-E2f03EMtEg.r09fBQqyqbI9-iSw8x2gla2TAuTBRpuEfBrlOyhiLMAg.PNG.handuelly/image.png?type%3Dw800)
​
0.01이 아니라 매우 작은 값이라면 무엇이든 사용 가능하다.  

Leaky ReLU는 x가 음수인 영역의 값에 대해 미분값이 0이 되지 않는다는 점을 제외하면 ReLU의 특성을 동일하게 갖는다.  

## PReLU

Leaky ReLU와 거의 유사하지만 새로운 파라미터 α 를 추가해 x가 음수인 영역에서도 기울기를 학습한다.  


PReLU 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfNyAg/MDAxNTgyNjA4ODUxNjQx.9XG-_nZCWmhVCmEBOfjn0PG1GB8SrDwKHdH9SO14fkQg.3_G-Nq3avOfDQkHbcj9Z6NbLaU4dePOW9cuF9PK4CdQg.PNG.handuelly/image.png?type%3Dw800)

ReLU 계열의 그래프
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTAw/MDAxNTgyNjA4OTgzNTY4.XGvKlH95zJAajkNhenW2hcntQh08wZ6hVG-1st3o1GAg.X6hCOdvIifvOzdULNFRLEOVw32J86leya7aUhLGNzYcg.PNG.handuelly/image.png?type%3Dw800)

## ELU

Exponential Linear Unit은 ReLU의 모든 장점을 포함하며 Dying ReLU 문제를 해결했다.  

출력 값이 거의 zero-centered에 가까우며, 일반적인 ReLU와 다르게 exp 함수를 계산하는 비용이 발생한다.  

ELU 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTQ4/MDAxNTgyNjA5MTM0Nzg2.cdhtmcFUhGVODjabbGIKclGA4r8x0wbMbPhu8le0ozYg.X9hpEkEqiciKoUTieHZzgs4QzrgdK6RJciLd3s89WnEg.PNG.handuelly/image.png?type%3Dw800)

![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMjQ2/MDAxNTgyNjA5MTkwNjE5.0Ybb5mPVJAoobiH5aVyWjXOp8a0umersu10id0WQWhAg.JR-wTuj4aFrk9fuMqQ39VPcUFu-TOqQHRuaWi2GCVBAg.PNG.handuelly/image.png?type%3Dw800)

## Maxout

ReLU의 장점을 모두 갖고, Dying ReLU 문제 또한 해결한다.  

하지만 계산해야 하는 양이 많고 복잡하다는 단점이 있다.  

Maxout 정의
![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfOTEg/MDAxNTgyNjA5MjQ4NDk0.vD-S64AEOzSpY_z0eiKdhcP11_eq6SOvuEVTcdPsOfkg.PeMj33JnGGId3oYIKoPe1adVORqX4Y6JegwwMjLBtcIg.PNG.handuelly/image.png?type%3Dw800)

# 비교

![](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfMTQ3/MDAxNTgyNjA5NDY3MTY3.228bUv_5mrol1w7X0NiFMD1UNru9zyf3yIJGcON-An0g.3Kzynlja9y_F9yTfANl937elQAK1pTGoJ_al7Om7TYsg.PNG.handuelly/image.png?type%3Dw800)

이처럼 매우 다양한 활성화 함수가 존재하는데, 각각마다 특징이 존재하기 때문에 상황에 맞춰 사용해야 한다.  

대표적으로 가장 많이 사용하는 함수는 ReLU인데, 이유는 간단하면서 사용이 쉽기 때문이다.  

sigmoid와 tanh는 사용해도 성능이 그닥 좋지 않다고 한다.  