# HRnet_estimator

<img src="https://github.com/ylab604/HRnet_BMI_estimator/blob/main/ming_0411_did.PNG">

-- 😈고려사항😈 --
1. 현재 batchsize =32 으로 바꿈
2. Hrnet-w18 로 세팅되어있음

-- 🐹Tory did🐹 --
1. validation loader 추가
2. 두번째 HRnet input channel 1x1 conv로 수정 (input channel 9로 조정)
3. 주석달기
4. 앞쪽 hrnet pth파일 업로드 


-- 🐹Tory to do list🐹 --
1. 이미지 배경제거
2. 논문읽기 
3. 데이터셋 보강 3d to Dexa

========================

-- 🥰ming did🥰 --
1. train_data : 2917개 / test_data : 1254개
2. DataLoader 구성
3. 2번째 HRnet Regression 모델 구성
4. Inference.py 구성


-- 🥰ming to do list🥰 --
1. Scaler 검토(datasets/BMI.py 수정)
2. minmaxscaler 저장(re-normalization을 위해서)
3. Attention 공부
4. regression head 구성
5. datasets/hanchaedae.py 만들기

========================

-- 😈주의사항😈 --
* yaml 파일 한글 주석처리해도 실행안됨


-- 👨‍👧‍👧Relate Work👩‍👧‍👦 --

https://github.com/ylab604/3D-human-body-paper-review
