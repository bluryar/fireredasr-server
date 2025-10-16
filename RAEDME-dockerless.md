# 安装

## DCU

```bash
cd <fireredasr-server>

uv venv --python 3.10

source .venv/bin/activate

uv pip install server/requirements-dcu.txt
```

## GPU

```bash
cd <fireredasr-server>

uv venv --python 3.10

source .venv/bin/activate

uv pip install server/requirements.txt
```

## 推理

1. 设置模型目录

```bash
/root/fireredasr-server/models
├── FireRedASR-AED-L -> /public/home/scn8xdhdqp/SothisAI/model/ExternalSource/FireRedASR-AED-L/main/FireRedASR-AED-L
└── PUNC-BERT -> /public/home/scn8xdhdqp/SothisAI/model/ExternalSource/FireRedChat-punc/main/FireRedChat-punc
    └── chinese-lert-base -> /public/home/scn8xdhdqp/SothisAI/model/ExternalSource/chinese-lert-base/main/chinese-lert-base
```

OR

```bash
cd server
mkdir -p models
# Download FireRedASR-AED-L & PUNC-BERT model and place it in models/
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
pushd models/PUNC-BERT && git lfs pull && popd
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L
pushd models/FireRedASR-AED-L && git lfs pull && popd
```

2. 运行

```bash
cd server
FIREREDASR_PATH=/root/fireredasr-server/FireRedASR MODEL_DIR=/root/fireredasr-server/models uvicorn src.main:app --host 0.0.0.0 --port 8000
```

- FIREREDASR_PATH: fireredasr 的项目根目录： `git clone https://github.com/FireRedTeam/FireRedASR.git`
- MODEL_DIR: 上面构建的模型目录
