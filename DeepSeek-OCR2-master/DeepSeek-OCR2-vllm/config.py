from pathlib import Path

# Project root is 3 levels up from this config file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:6
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
# MODEL_PATH = 'deepseek-ai/DeepSeek-OCR-2' # change to your model path
MODEL_PATH = "/home/bo/models/deepseek-ocr-2"

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py



# INPUT_PATH = '/your/image/path/'
# OUTPUT_PATH = '/your/output/path/'

# INPUT_PATH = str(PROJECT_ROOT / 'input'/ 'npl.png') # image path
# INPUT_PATH = str(PROJECT_ROOT / 'input'/ '惠元2024年第七期不良资产证券化信托受托机构报告2026年度第1期（总第16期）.pdf') # pdf path
# INPUT_PATH = str(PROJECT_ROOT / 'input'/ '建欣2025年第十八期不良资产支持证券评级报告及跟踪评级安排（中债资信）.pdf') # pdf path
INPUT_PATH = str(PROJECT_ROOT / 'input') # Bo: batch run for all the pdf in the input folder
OUTPUT_PATH = str(PROJECT_ROOT / 'output')

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
