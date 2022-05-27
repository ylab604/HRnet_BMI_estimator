# HRnet_estimator

<img src="https://github.com/ylab604/HRnet_BMI_estimator/blob/main/ming_0411_did.PNG">

========================

전승진 바보 🤪 이걸 며칠만에 볼까?

-- ☠**주의사항**☠ --
* yaml 파일 한글 주석처리해도 실행안됨


-- 😈**고려사항**😈 --
1. 현재 batchsize =16 으로 바꿈
2. model_DN : Hrnet-w18-v2 
3. model_reg : Hrnet-w18-C // Hrnet-w48-C
4. 2D to BMI : train_data : 2917개 / test_data : 1254개

========================

-- 🐿**Tory did**🐿 --
1. validation loader 추가
2. 두번째 HRnet input channel 1x1 conv로 수정 (input channel 9로 조정)
3. 주석달기
4. 앞쪽 hrnet pth파일 업로드 
5. 이미지 배경제거
6. 데이터 증강

-- 🐹**Tory to do list**🐹 --
1. 논문읽기 
2. 데이터셋 보강 3d to Dexa

========================

-- 🤩**ming did**🤩 --
1. DataLoader 구성
2. 2번째 HRnet Regression 모델 구성
3. Inference.py 구성
4. Scaler 검토(datasets/BMI.py 수정)
5. minmaxscaler 저장(re-normalization을 위해서)


-- 🥰**ming to do list**🥰 --
1. Attention 공부
2. datasets/hanchaedae.py 만들기
3. 논문읽기
4. 데이터 셋 정리

========================

-- **현재 수행해야 하는 것** --
<img src="https://github.com/ylab604/HRnet_BMI_estimator/blob/main/image/0413_hanchaedae_todo_framework.jpg">

-- 👨‍👧‍👧**Relate Work**👩‍👧‍👦 --

https://github.com/ylab604/3D-human-body-paper-review
