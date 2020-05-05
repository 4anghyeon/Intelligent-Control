# Intelligent-Control
PyTorch를 이용한 지능제어 텀프로젝트
## 목표
주어진 4가지 클래스의 Wi-Fi 수신 신호 패턴 데이터셋을 이용해 실내 사람 수 예측
### 클래스 조건
신호 데이터 하나 당 3x3x56 개의 크기 값<br>
![Image01](https://pgx5sg.ch.files.1drv.com/y4mANDlruuv1VSdFM4-F_peWQsNkWrroLMKbgA9CxMrLinGa91wpaAy_h4UAJtSrN7__I7rVkm8c5ByRBviyz23SuylWJ7YFmnmZGbuzqKVQCPJ6EKq_qtxQmmZSrenhJCu1OGhJAJA1IOeXJr3AgD3bOWSmG5b6E2iPuQgNY9BLPuI7LvsPC62ZeA4OhoJogjXbRfzFVx_yFnfIua2TxBkVw?width=500&height=269&cropmode=none)<br>
![Image02](https://pwx5sg.ch.files.1drv.com/y4mhTaWArZVBl79bTkPjG_lYLxvZ2hmibmY8qjeDbYtxxJLkArkxQTSEiv6QLZzlDBegAhV0kfRtoUiP2DUZLDrPn34xg67KNkYgVybe7aNRl_zEZxJEMiiDbl21yKwG9FrLZPTPpSNG5BGaocd7kR19EXDuXFHW3oeUGymH0cUU9Ula0HueW8aUqnuNZdihAMHP6GzMbQ9wnmOCL9DH1eIlA?width=500&height=319&cropmode=none)<br>
### MLP 기본 설계
* P0W0 부터 P14W0 까지의 데이터셋을 전부 합치면 15039X56 크기의 데이터가 만들어짐
* 이 중 50%를 학습데이터로 나머지 50%를 테스트 데이터로 활용함 -> 각 7519x56의 크기
* 설계 요구사항의 epoch = 50
* 설정한 batch size = 500
* 7569/500 ≒ 15 이므로 15 iteration = 1 epoch
* 입력 크기는 56, 출력 크기는 4를 가져야 하므로 다음과 같은 기본 3층 신경망 구성
![Image02](https://pmx5sg.ch.files.1drv.com/y4mlBKuA6EdBumz2bOrZ5b44DAiWZVdZ6RlR_nZDP4dFGlvXH6KGU2ViAZ3PswWp0eV6l60O47atW4H8K39Ihl9ZoYLEftOIpvcf0yQZny6JInAo5ZoMH-cYZ6WLSFc7PP-I8ytBywvQ3q6zS2JHF2SeayLy7RlhyuBKWZdweHECXcQ26xKnWsjzXQQbkIgr7WjAjkHn3VGbgOfknnJXTBU9w?width=519&height=105&cropmode=none)<br>

### MLP 결론
* 활성화 함수: ReLU, Leakyrelu
* 신경망 개수: 5개 (56,50) -> (50,90) -> (90,75) -> (75,30) -> (30,4)
* 옵티마이저: Adagrad
* 손실함수: Softmax
* 모델: 1층 신경망 -> LeakyRelu -> 2층 신경망 -> ReLU -> 3층 신경망 ->  ReLU -> 4층 신경망 -> LeakyRelu -> 5층 신경망
* 학습결과<br>
![Image03](https://p2x5sg.ch.files.1drv.com/y4ms61c0dMOGD9FfrCDNDrARcdYpECQBG8lVx3yLZPNzBS-z7aCvHJDenVOy5gjRG0mrRf7Fy0ISsZBUIp4qvPhG6XLqJOTeOlwBX5T-f_sptk8KJVi1ROQwrSPgTfRghOTHpQOJsnTbTqEB8rDQsB2CDKY92mFlMUBzcANHiTGNBgGsuJstmXFVpAN9mPCYhjwig0izD53qjrp8Cw8jHI1pQ?width=475&height=280&cropmode=none)<br>
* 최종 cost = 0.07
* 정확도 = 0.945
* Epoch = 50으로 고정되어 있어 더 많은 학습을 하기 위해 batch size를 60으로 줄였더니 정확도가 많이 늘어났다.

### CNN 기본 설계
* P0W0 부터 P14W0 까지의 데이터셋을 전부 합치면 15039X56 크기의 데이터가 만들어짐
* 이 중 50%를 학습데이터로 나머지 50%를 테스트 데이터로 활용함 -> 각 7519x56의 크기
* 이를 batch size =500의 크기로 (500,1,8,7)의 Tensor(500개의 1채널의 8x7크기의 데이터)로 변환하였다.
* 기본으로 2개의 conv layer(layer1: 3x3의 32개의 필터, stride=1, padding=1 -> ReLu -> 2x2,stride=2로 Maxpool, 
* layer2: 3x3의 64개의 필터, stride=1, padding=1 -> ReLu -> 2x2,stride=2로 Maxpool)인 신경망을 구성하였다.

### CNN 결론
* 레이어 개수: 4개

* Layer1: Conv2d(1, 32, kernel_size=3, stride=1, padding=1) -> ReLU() 
	->MaxPool2d(kernel_size=2, stride=2)
     Layer2:Conv2d(32, 64, kernel_size=3, stride=1, padding=1) -> ReLU()
	->torch.nn.MaxPool2d(kernel_size=2, stride=2))
       Layer3:Conv2d(64, 92, kernel_size=3, stride=1, padding=1) -> ReLU()
	->torch.nn.MaxPool2d(kernel_size=2, stride=2))
	Layer4:ReLU() -> Dropout(n=0.2)
* 옵티마이저: Adam

* 손실함수: Softmax
