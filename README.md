# HRnet_estimator

<img src="https://github.com/ylab604/HRnet_BMI_estimator/blob/main/ming_0411_did.PNG">

-- 고려사항 --
1. 현재 batchsize =32 으로 바꿈
2. Hrnet-w18 로 세팅되어있음



-- to do --
1. validation loader 추가
2. ~~두번째 HRnet input channel 1x1 conv로 수정~~ (input channel 9로 조정)
3. 주석달기
4. 앞쪽 hrnet pth파일 업로드 



-- 주의사항 --
* yaml 파일 한글 주석처리해도 실행안됨


-- Relate Work --

https://github.com/ylab604/3D-human-body-paper-review


-- ming did --
1. train_data : 2917개 / test_data : 1254개
2. DataLoader 구성
3. 2번째 HRnet Regression 모델 구성


--ming to do list--
1. Inference.py 구성
2. Scaler 검토
3. Attention 공부
4. regression head 구성
