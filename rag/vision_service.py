"""
视觉分析服务：调用 Qwen-VL 视觉大模型识别图片中的违法行为，可结合 RAG 判定标准。
"""
from langchain_core.messages import HumanMessage
from model.factory import vision_model
from utils.logger_handler import logger

# 用户上传图片缓存：img_id -> data URI（base64）
_uploaded_images: dict[str, str] = {}


def register_uploaded_image(img_id: str, data_uri: str) -> None:
    """注册用户上传的图片，供 vision_analyze 工具按图片ID引用。"""
    _uploaded_images[img_id] = data_uri


def resolve_image(image: str) -> str:
    """把图片ID解析为data URI；URL或data URI则原样返回。"""
    if image.startswith("img_"):
        return _uploaded_images.get(image, "")
    return image


VISION_PROMPT_TEXT = """你是道路/现场取证图像分析专家。请基于图片内容识别违法行为类型，结合给定的判定标准与参考案例输出专业、严谨、可落地的分析结果。

{prompt_body}

输出要求：
1. 先给出识别结论（行为类型/是否属于道路养护），并说明置信度；
2. 列出判断依据（图片可见要素、检测框位置、时间线索等）；
3. 若结合了判定标准/参考案例，说明图片情况与标准/案例的对应关系；
4. 不确定或图片信息不足时，明确说明缺失项与补证建议，不得臆断；
5. 涉及法律定性时提示"需人工/执法部门复核"。"""


class VisionService:
    def __init__(self):
        self.model = vision_model

    def analyze(self, image: str, question: str, context: str = "") -> str:
        image_ref = resolve_image(image)
        if not image_ref:
            return f"未找到图片 {image}：请确认已通过聊天输入框真实上传图片（不要手动输入图片ID），上传后重新提问。"

        parts = []
        if context and context.strip():
            parts.append(f"【RAG判定标准与参考案例】\n{context}")
        parts.append(f"【用户问题】\n{question}")
        prompt = VISION_PROMPT_TEXT.format(prompt_body="\n\n".join(parts))

        try:
            response = self.model.invoke([
                HumanMessage(content=[
                    {"image": image_ref},
                    {"text": prompt},
                ]),
            ])
            content = response.content
            return content if isinstance(content, str) else str(content)
        except Exception as e:
            logger.error(f"[vision_analyze]视觉模型调用失败：{e}", exc_info=True)
            return f"视觉模型调用失败：{e}"


if __name__ == '__main__':
    vs = VisionService()
    print(vs.analyze(
        image="data:image/jpeg;base64,AAAA",  # 无效示例，仅用于展示调用方式
        question="这张图片是什么违法行为？",
    ))