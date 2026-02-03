import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr2 import DeepseekOCR2ForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids= {128821, 128822})]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """pdf2images"""
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, pdf_output_dir):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{pdf_output_dir}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, pdf_output_dir):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, pdf_output_dir)
    return result_image


def process_single_image(image):
    """single image"""
    prompt_in = PROMPT
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCR2Processor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


def get_pdf_files(input_path):
    """Get all PDF files from input directory"""
    input_dir = Path(input_path)
    pdf_files = list(input_dir.glob("*.pdf"))
    pdf_files.sort()
    return pdf_files


def process_single_pdf(pdf_path, output_base_path):
    """Process a single PDF file and save outputs"""
    pdf_name = pdf_path.stem
    pdf_output_dir = Path(output_base_path) / pdf_name

    os.makedirs(pdf_output_dir, exist_ok=True)
    os.makedirs(pdf_output_dir / 'images', exist_ok=True)

    print(f'{Colors.BLUE}Loading PDF: {pdf_path.name}{Colors.RESET}')

    images = pdf_to_images_high_quality(str(pdf_path))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc=f"Pre-processing {pdf_name}"
        ))

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )

    mmd_det_path = pdf_output_dir / f'{pdf_name}_det.mmd'
    mmd_path = pdf_output_dir / f'{pdf_name}.mmd'
    pdf_out_path = pdf_output_dir / f'{pdf_name}_layouts.pdf'

    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0

    for output, img in zip(outputs_list, images):
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                continue

        page_num = '\n<--- Page Split --->'

        contents_det += content + f'\n{page_num}\n'

        image_draw = img.copy()

        matches_ref, matches_images, mathes_other = re_match(content)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, str(pdf_output_dir))

        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(mathes_other):
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'

        jdx += 1

    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, str(pdf_out_path))

    print(f'{Colors.GREEN}Completed: {pdf_name}{Colors.RESET}')
    return pdf_name


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    pdf_files = get_pdf_files(INPUT_PATH)

    if not pdf_files:
        print(f'{Colors.RED}No PDF files found in {INPUT_PATH}{Colors.RESET}')
        exit(1)

    print(f'{Colors.YELLOW}Found {len(pdf_files)} PDF files to process{Colors.RESET}')

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f'\n{Colors.YELLOW}[{i}/{len(pdf_files)}] Processing: {pdf_file.name}{Colors.RESET}')
        try:
            process_single_pdf(pdf_file, OUTPUT_PATH)
        except Exception as e:
            print(f'{Colors.RED}Error processing {pdf_file.name}: {e}{Colors.RESET}')
            continue

    print(f'\n{Colors.GREEN}All PDFs processed. Output saved to: {OUTPUT_PATH}{Colors.RESET}')

    # Cleanup: destroy distributed process group to avoid NCCL warning
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
