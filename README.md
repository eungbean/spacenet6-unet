# 여러분께 남기는 노트

* 기존 코드에서 관련없는 부분은 파일/폴더명 앞에 ```_``` 처리를 해놨습니다. 
* 코드는 [motokimura](https://github.com/motokimura/spacenet6_solution)의 코드를 베이스로 하고 있습니다.
* 상당부분은 바로 train이 돌아가도록 미리 세팅해놨습니다.
* model 구현에 주력해주세요.

## U-Net + EfficientNet (v0)  (원래 챌린지 코드)
### Models
```
spacenet6_model/models/__init__.py
```
* ```get_unet``` 함수에서 직접 U-Net을 불러옵니다.

### Train Code
```
tools/train_spacenet6_model.py
./1_train_unet_v0.sh
```


## U-Net + DoubleConv 코드 (v1) (가장 간단한 U-Net)
### Models
```
spacenet6_model/models/unet/__init__.py
spacenet6_model/models/unet/unet_model.py
spacenet6_model/models/unet/unet_parts.py
```

원본 코드를 어떤 형태로 가공했는지 (```unet_model.py```) 유심히 살펴보세요.   
가장 중요한 파트입니다.

### Train Code
```
tools/train_unet_v1.py
./1_train_unet_v1.sh
```

## 여러분이 구현할 U-Net 코드 (v2)

### Models
```
spacenet6_model/models/unet_ours/*
```

현재는 같은 코드로 채워져있습니다.  
이 곳에 코딩을 하게 될 겁니다.

### Train
```
tools/train_unet_v2.py
./1_train_unet_v2.sh
```


### Train 하는 법

* Train Unet_v0 (EfficientNet)
```
./1_train_v0.sh
```

* Train Unet_v1 (DoubleConv)
```
./1_train_v1.sh
```

* Train Unet_ours
```
./1_train_v2.sh
```

### 진행상황 보는 법
```
./train_exp_{0000}.out
```

### 실험중단 하는법
```
./0_stop_train.sh
```