---
title: "다층 퍼셉트론과 XOR gate 구현"

categories:
  - DL
tags:
  - [Deep Learning, perceptron]

toc: true
toc_sticky: true

date: 2022-07-13
last_modified_at: 2022-07-13
---

# 1. 단층 퍼셉트론의 한계

## XOR GATE

**AND gate 진리표**

|$$x_1$$  |$$x_2$$  |$$y$$  |
|----|----|----|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

XOR게이트는 배타적 논리합이라고 하는 논리 회로로, 입력값 $$x_1, x_2$$ 중 하나만 1일 때 1을 출력한다.  
하지만, 단순 퍼셉트론으로 XOR 게이트를 구현하는 것은 불가능하다.  
이유는 XOR 논리회로의 경우, 각 입력값의 조합 ($$x_1, x_2$$)을 한 직선으로 구분지을 수 없기 때문이다.

[퍼셉트론 논리식]

$$y = \begin{cases}
0, \;if\;​(b + w_1​x_1​ + w_2​x_2 \leq θ)\\ 
1, \;if\;(b + w_1​x_1 ​+ w_2​x_2​ > θ)
\end{cases}$$

XOR 게이트의 경우에는 각 조합에 따른 출력값들을 하나의 직선으로 나누는 게 불가능하다.  
다음 그림을 보면 이해가 쉽다.
![](https://velog.velcdn.com/images/citizenyves/post/9594d7df-ad9f-47d0-9ee5-7646ad8e9834/image.png)

## 선형 및 비선형
직선이라는 조건을 없앤다면 퍼셉트론의 XOR 게이트는 가능하다.  
예를 들어, 다음 그림과 같이 곡선을 이용하는 방법이 있다.  
참고로 곡선의 영역을 비선형 영역, 직선의 영역은 선형 영역이라고 하며 아래 그림은 비선형 영역이라고 할 수 있다.  
![](https://velog.velcdn.com/images/citizenyves/post/befaf149-5583-417e-acdc-a3e80687576b/image.png)

# 2. 다층 퍼셉트론

## 퍼셉트론의 층 쌓기

단층 퍼셉트론으로(하나의 직선) XOR 논리를 만족할 수 없었지만, 비선형 영역 개념을 이용하여 XOR 진리표에 따른 출력값들을 정확히 나눌 수 있었다.  
퍼셉트론은 층을 쌓아 다층 퍼셉트론(Multi-Layer Perceptron, MLP)을 만들 수 있는데, 이 방법으로 XOR을 표현할 수 있다.  

XOR 게이트를 만드는 방법은 다양하지만, AND, NAND, OR 게이트를 서로 조합하는 방식을 선택했다.  


위 그림과 같이 각 게이트를 한번씩 사용하면서 총 2개의 layer를 사용했다.  
$$x_1, x_2$$가 NAND와 OR 게이트의 입력이 되고, NAND와 OR 게이트의 출력이 AND 게이트의 입력으로 들어가는 구조이다.  
다음은 NAND으로부터의 출력을 $$S_1$$, OR의 출력을 $$S_2$$​로 하는 XOR 게이트의 진리표다.  

**XOR 게이트 진리표**

|$$(x_1, x_2)$$  |$$(S_1, S_2)$$  |$$y$$  |
|----|----|----|
| (0, 0) | (1, 0) | 0 |
| (1, 0) | (1, 1) | 1 |
| (0, 1) | (1, 1) | 1 |
| (1, 1) | (0, 1) | 0 |


첫번째 layer의 출력인 ($$S_1, S_2$$)를 두번째 layer(AND 게이트)에 입력하여 출력하면 XOR 게이트를 만족하는 값을 확인할 수 있다.  

## XOR 게이트 python 구현

```python
#AND GATE
def AND(x1, x2):
    x = np.array([x1, x2]) #입력값
    w = np.array([0.4, 0.4]) #가중치
    b = -0.7 #편향
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

#NAND GATE
def NAND(x1, x2):
    x = np.array([x1, x2]) #입력값
    w = np.array([-0.4, -0.4]) #가중치
    b = 0.7 #편향
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

#OR GATE
def OR(x1, x2):
    x = np.array([x1, x2]) #입력값
    w = np.array([0.4, 0.4]) #가중치
    b = -0.2 #편향
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

#XOR GATE
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, x2)
    return y

===============   
XOR(0, 0) #0출력
XOR(1, 0) #1출력
XOR(0, 1) #1출력
XOR(1, 1) #0출력
```

구현된 XOR을 뉴런을 활용한 퍼셉트론으로 표시한 그림은 다음과 같다.  
![](https://velog.velcdn.com/images/citizenyves/post/209b81fe-121e-43ea-a0f6-fd7e6698631b/image.png)

AND, OR, NAND의 경우 모두 단층 퍼셉트론이지만, XOR은 이제 2층으로 구성된 퍼셉트론인 것을 확인할 수 있으며 이를 다층 퍼셉트론(MLP)이라 한다. (0~2까지 총 3개의 층이지만, 가중치를 갖는 층을 기준으로 2층이라 한다)

> 정리  
> 1) 0층의 두개 뉴런: $$x_1, x_2$$(입력 신호)를 받아 1층 뉴런으로 신호를 보냄  
> 2) 1층의 두개 뉴런: 2층 뉴런으로 신호를 보냄  
> 3) 2층 뉴런: 최종값 y 출력