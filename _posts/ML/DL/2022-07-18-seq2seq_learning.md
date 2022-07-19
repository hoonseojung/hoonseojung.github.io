---
title: "seq2seq learning"

categories:
  - DL
tags:
  - [Deep Learning, seq2seq]

toc: true
toc_sticky: true

date: 2022-07-18
last_modified_at: 2022-07-19
---

# seq2seq learning

## seq2seq 모델이란?
- sequential 패턴들이 들어왔을 때 그것을 입력으로 받아 모두 처리한 후 다시 또다른 sequence를 output으로 출력해주는 구조
- input : sequence of items(단어, letter, feautres of an images 등)
- output : item의 또다른 sequence(Input과 개수 다를 수 있음)
- encoder와 decoder로 구성되어 있음(정보의 압축 & 생성)
- encoder는 입력된 정보를 어떻게 처리해서 저장할 것인가
    - input sequence의 item을 process하여 그것들이 가지고 있는 정보들을 compile하여 하나의 벡터(context vector)로 표현(정의)
    - context vector를 생성하게 되면(최종적으로 모든 Input sequence에 대한 정보들을 받아들여 처리하게 되면) encoder는 context vector를 decoder에게 넘겨줌
- decoder는 encoder로부터 압축된 정보를 어떻게 풀어서 반환해 줄 것인가
    - decoder는 encoder로부터 context vector를 받아서 output sequence를 item by item으로 만듦
- encoder-decoder의 구조를 표현하기 위한 가장 고전적인 방식으로는 Recurrent neural network(RNN)이 사용됨
    - RNN은 0 state의 hidden state(hidden state #0가 있을 때, 첫번째 sequence input이 들어오면 hidden state #0와 함께 item을 proccessing하여 hidden state #1과 output vector를 생성함
    - 위 과정을 앞으로 오는 input에 대해 반복함
    - 각 Input이 들어올 때마다 hidden state가 update가 되고, 최종적으로 모든 input이 들어온 다음에 마지막 hidden state가 context vector가 되어 decoder로 전달됨
    - encoder가 가지고 있는 context vector, 즉 RNN에서의 pure한 final state는 가장 뒤쪽의 item의 영향을 많이 받고, 앞쪽에 해당하는 item에 대해서는 영향을 거의 받지 못할 가능성이 굉장히 높음
        - 이 경향을 해결하기 위해 LSTM, GRU가 만들어져있었지만, long term에 대해서 완벽하게 해결해주는 것은 아님
        - 긴 문장(long sequence)에 대해서는 context vector가 bottle neck이 발생하기에 어려움이 있음
        - 그렇기 때문에 최근에는 Attention이라는 개념을 도입해 모델이 input sequence 중에서 현재 output item이 주목해야하는 파트들을 다이렉트로 커넥션을 주어(가중치를 주어서) 해당하는 파트의 부분들을 조금 더 활용할 수 있도록 해주는 것이 attention 구조
    - Bahdanau attention은 attention score 자체를 학습하는 neural network model이 존재
    - Luong attention은 attention score를 따로 training시키지 않고 현재 hidden state와 기존의 과거 hidden state들의 유사도를 측정하여 attention score를 만듦
    - 위 두 attention 사이의 퍼포먼스 차이가 그렇게까지 크진 않은 것으로 알려져 있음 —> 학습시키지 않고 단순한 곱셈 연산으로 계산하는 attention이 학습시키는 attention과 성능의 차이가 크지 않다면, 당연히 실용적으로 Luong을 사용

## attention mechanism이 적용된 seq2seq learning과 기존의 classic한 방법 차이점
    - encoder가 final hidden state뿐만 아니라 모든 hidden state를 decoder에게 넘겨줘 더 많은 정보를 넘겨줌
    - 그러면 decoding이 수행되는 과정에서 필요한 hidden state를 선택하여 서로 다른 가중치를 통해 활용함
    - decoder의 관점에선 encoder가 전달해주는 여러 hidden state를 가지고 output을 가지고 extra step을 수행하게 됨
        - extra step : encoder가 보내준 hidden state를 다 봄
        - 이 hidden state는 아무리 LSTM이나 GRU를 사용했어도 기본적인 가정은 각각의 hidden state는 해당하는 단어의 sequence에 가장 영향을 많이 받음
        - 현재 output을 생성하고자 하는 단계에서의 hidden state들의 score를 만들고, score에 대해서 softmax를 수행한 후 해당하는 값들을 모두 결합하여(score가 클수록 중요도 큼) 하나의 weighted vector 생성 = 해당 time step의 context vector
            - score —> Luong attention의 경우 각각의 hidden state에 score를 매기는데, score 자체가 decoder와 encoder의 hidden state를 내적한 값이라 봐도 무방
        - 그렇게 만들어진 context vector와 decoder의 hidden state vector를 결합(concatenate)하여 neural network의 입력으로 넣으면 최종적으로 output vector가 토큰으로 반환됨

## 히트맵
왼쪽 열 = 실제 입력 값, 오른 행 = 번역 출력 값, 각각의 입력 값이 출력된 번역 값에 어디와 attention이 많이 되어있는지, 밝을 수록 많이 된 것