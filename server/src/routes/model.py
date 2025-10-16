import os
import sys
import re
import traceback
from collections import OrderedDict

# Add FireRedASR to Python path
sys.path.append(os.getenv('FIREREDASR_PATH'))

from src.routes.fireredasr import FireRedAsr, FireRedAsrConfig
from redpost.models.redpost import RedPost, RedPostConfig

from fastapi import File, UploadFile, APIRouter
from fastapi.responses import JSONResponse

from src.helpers import load_audio

# Create FastAPI router
router = APIRouter(
    tags=["Model"],
)

# Initialize model
model = None
post_model = None

def get_model():
    global model, post_model
    if model is None:
        model_dir = os.path.join(os.getenv('MODEL_DIR'), "FireRedASR-AED-L")
        asr_config = FireRedAsrConfig(
            use_gpu=True,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=False,
            use_half=True
        )
        model = FireRedAsr.from_pretrained("aed", model_dir, asr_config)

    if post_model is None:
        # POST (PUNC+ITN)
        punc_model_dir = os.path.join(os.getenv('MODEL_DIR'), "PUNC-BERT")
        post_config = RedPostConfig(
            use_gpu=True,
            sentence_max_length=30
        )
        post_model = RedPost.from_pretrained(punc_model_dir, post_config)

    return model, post_model

@router.post("/audio/transcriptions", status_code=200)
async def transcribe_audio(
    file: UploadFile = File(...)
):
    """
    Transcribe audio with FireRedASR model.
    """
    try:
        # Load and prepare audio
        audio = load_audio(file)
        
        # Get model instance
        model, post_model = get_model()
        
        # Perform transcription
        outputs = model.transcribe(["test_audio"], [(16000, audio)])

        asr_results = OrderedDict({r["uttid"] : r for r in outputs})

        # Filter ASR Results
        new_asr_results = OrderedDict()
        text = ""
        lowest_confidence = 1.0
        # 新增：一个标志位，用于追踪是否有任何结果包含置信度
        has_any_confidence = False

        for k, r in asr_results.items():
            # 修复：检查 'confidence' 键是否存在
            if "confidence" in r:
                has_any_confidence = True
                if r["confidence"] < lowest_confidence:
                    lowest_confidence = r["confidence"]
            
            t = r["text"].lower()
            if re.search(r"<blank>", t):
                print("BLANK:", r)
                continue
            if re.search(r"^[哎嘿哈呀嘟呐啊呵呃哒啦嗯呜噔嘣咕嘀滴哦喔嗡咚嘞哇]+$", t):
                continue
            if re.search(r"^(oh |ooh |oooh |doo |do |na |la |da |biu |du |wu |em |bae |ah |yeah |wow |bada |dum )+$", t + " "):
                continue
            t = re.sub(r"([哎嘿哈呀嘟呐啊呵呃哒啦嗯呜噔嘣咕嘀滴哦喔嗡咚嘞哇]){4,}", "", t)
            r["text"] = t
            text += t
            new_asr_results[k] = r
        asr_results = new_asr_results

        # POST
        # add extra context to facilitate end of sentence punctuation
        text = text + "你好呀"
        batch_post_results = post_model.process([text], ["text"])
        text = "".join([r["punc_text"] for r in batch_post_results])
        text = re.sub("<unk>|<UNK>|\[unk\]|\[UNK\]", "", text)
        text = text[:-4]
        if len(text) > 0:
            if text[-1] == ",":
                text = text[:-1] + "."
            if text[-1] == "，":
                text = text[:-1] + "。"

        # Format response
        # 先构建一个不包含 confidence 的响应
        response_content = {
            "sentences" : [{
                "text": text,
            }],
            "wav_file" : file.filename
        }

        # 修复：只有当至少有一个结果包含置信度时，才在响应中添加 confidence
        if has_any_confidence:
            response_content["sentences"][0]["confidence"] = lowest_confidence
        
        return JSONResponse(content=response_content)
        
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
