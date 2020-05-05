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
