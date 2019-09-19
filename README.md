# Universal-Adversarial-Text

## 1. 설명

<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/adv_text_purpose.png" width=100%><br>
</div>

- 주어진 NLP 모델에 대해 **단 하나의 perturbation 문장을 생성함**으로써 모델을 속이는 **Universal Adversarial Attack** 기법
- Universal adversarial attack에 의해 생성되는 문장은 입력 문장에 추가되고, 이로 인해 문장의 분류 결과는 기존과 다르게 나타남
- 제안 기법은 baseline 기법에 비해 약간 좋은 성능을 나타내고, 더욱 효율적으로 attack을 수행할 수 있음

## 2. 모델 구조

<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/algorithm.png" width=80%><br>
<br>
</div>

**(a) Original dataset을 이용하여 text classification 모델 학습**
  - 분류 모델 : **embedding layer + 2 layer LSTM cell + FCN**
  - e (word embedding 차원) = 300
  - LSTM cell의 hidden node 수 = 128
  - FCN의 hidden node 수 = 256
   
**(b) Adversarial text의 word embedding 값 최적화**
  - 특정 문장에 대해 embedding layer의 output인 **embedding 값과**, random initialize된 **길이 n의 adversarial text의 word embedding 값을 concatenate**
  - 이후 output이 바뀌도록 하는 adversarial text의 embedding 값을 optimize
  - 이전의 문장 분류 모델의 weight은 고정
   
**(c) Adversarial text의 각 단어 추출**
  - adversarial text의 embedding 값 * embedding layer의 lookup table 의 가장 높은 column index
  - adversarial text의 embedding 값과 lookup table 사이 **cosine similarity**를 계산하는 방식
  - **추출된 문장을 원본 문장의 뒤에 concatenate** 하여 adversarial example 생성, 테스트 수행
  
## 3. 실험
- Dataset
  - [IMDB review dataset](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset)
  - 22,500 training data + 2,500 validation data + 25,000 test data
- Training model
  - epoch = 20, learning rate = 0.001, batch size = 250, Adam optimizer
- Optimize adversarial text
  - epoch = 10, learning rate = 0.0001, batch size = 250, Adam optimizer
- Baseline model  
  - [TextFool](https://github.com/bogdan-kulynych/textfool)
    
## 4. 결과
- Qualitative result

<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/results.png" width=80%><br>
</div>

  - Baseline과 거의 **유사한 성능**으로, 하지만 **더욱 빠르게** 실행 가능
    
- Quantitative result
  - Adversarial text
    
<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/adv_text.png" width=80%><br>
</div>

  - 자연스러운 문장은 아님 : Language model을 고려하지 않음
      
## 5. 고찰
- Adversarial text의 각 word의 embedding 값의 범위 초과
  
<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/adv_weights.png" width=85%><br>
<br>
</div>  

  - Optimized adversarial text의 embedding 값이 **기존 word embedding 값의 범위를 초과**함
  - 학습되는 parameter의 weight을 제한하는 기법이 필요함
  
- 자연스럽지 못한 adversarial text 문장
  - (앞서 말한) 범위 제약 조건의 부재
  - Adversarial text 내 각 word를 독립적으로 optimize하기 때문
  - Language model의 속성을 고려해야 함
  - **k-Nearest Neighbor 기법 + beam search** 적용
    
<div align="center">
 <img src="https://github.com/pod3275/Universal-Adversarial-Text/blob/master/assets/beamsearch.png" width=100%><br>
</div>    
      
## 6. 결론
- 하나의 모델에 대해 단 하나의 adversarial text를 생성하는 Universal adversarial attack을 NLP 모델에의 적용 연구
- 실험 결과 제안 기법은 성능이 비슷한 baseline 모델에 비해 더욱 **시간 효율적**으로 attack 수행 가능
- Text 데이터는 continuous한 image 데이터와는 달리 **discrete**하기 때문에, 이 특징을 바탕으로 각 input sentence의 동일한 위치의 단어에 대한 modification 혹은 동일한 위치에 특정 단어를 insertion하는 등의 작업을 수행하는 universal adversarial attack 가능
  
