import os

import pandas as pd
import json
import re
import ast


csv_path = "data/external/records.csv"

md_path = "data/违法案例库.md"

metadata_path = "data/违法案例库_metadata.json"


# ==============================
# 提取模型JSON结果
# ==============================

def parse_model_output(text):

    # --- 内部定义一个专门用来提取和解析 JSON 的小方法 ---
    def parse_text(text):
        # 1. 截断：如果有 </think>，直接把包含 </think> 及以前的所有废话全部切掉
        if '</think>' in text:
            text = text.split('</think>')[-1]

        # 2. 寻找 ```json ... ``` 或 ``` ... ``` 里的内容
        block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
        if block_match:
            json_str = block_match.group(1)
        else:
            # 如果没有代码块，再去找大括号 {...}（使用贪婪匹配 .* 获取完整的最外层括号）
            match = re.search(r'\{.*\}', text, re.S)
            if match:
                json_str = match.group(0)
            else:
                return None

        # 3. 清理注释
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.S)  # 去掉块注释
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.M)  # 去掉行注释

        # 4. 尝试解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("标准 JSON 解析失败，尝试使用 ast 解析...")
            try:
                # 把可能误伤的 ... 替换为 []，防止 ast 将其解析为 Ellipsis 对象
                json_str_safe = json_str.replace("...", "[]")
                return ast.literal_eval(json_str_safe)
            except Exception as e:
                print(f"⚠️ ast 解析也失败: {e}, 原始数据: {json_str}")
        return None

    # --------------------------------------------------

    if isinstance(text, dict):
        print("response 类型: dict")
        result_data = text

    elif isinstance(text, str):
        print("response 类型: str")
        parsed = parse_text(text)
        if parsed is not None:
            result_data = parsed

    elif isinstance(text, list) and text and isinstance(text[0], str):
        print("response 类型: list[str]")
        parsed = parse_text(text[0])
        if parsed is not None:
            result_data = parsed

    else:
        print("response 类型未知:", type(text))

    return result_data

def clean_model_output(text):
    """
    清理模型输出中的 Markdown 格式符号，
    保留实际文本内容。
    """

    if text is None:
        return ""

    text = str(text)

    # 1. 删除 Markdown 代码块标记
    text = re.sub(r"```(?:json|JSON|text|txt)?", "", text)
    text = text.replace("```", "")

    # 2. 删除标题符号
    # ### Analysis Process -> Analysis Process
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)

    # 3. 删除加粗、斜体符号
    text = text.replace("**", "")
    text = text.replace("__", "")

    # 4. 删除 Markdown 删除线
    text = text.replace("~~", "")
    text = text.replace("-", "")

    # 5. 删除 Markdown 引用符号
    text = re.sub(r"(?m)^\s*>\s?", "", text)

    # 6. 删除 Markdown 列表符号
    # - xxx
    # * xxx
    # + xxx
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)

    # 7. 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. 去掉首尾空白
    text = text.strip()

    return text

# ==============================
# 读取CSV
# ==============================

df = pd.read_csv(
    csv_path,
    encoding="utf-8"
)


markdown_list = []

metadata_list = []



for _, row in df.iterrows():


    # --------------------------
    # 提取模型结果
    # --------------------------

    result_json = parse_model_output(
        row["model_output"]
    )


    if result_json:


        result = result_json.get(
            "result",
            ""
        )


        boxes = result_json.get(
            "bounding_boxes",
            []
        )


    else:

        result = ""

        boxes = []



    # --------------------------
    # 生成RAG文本
    # --------------------------

    md = f"""
案例编号：{row['id']}

违法类型：
{row['违法类型']}

违法描述：
{row['TJ_NAME']}

发生地点：
{row['发生地点']}

发生时间：
{row['发生时间']}

所属支队：
{row['所属支队']}

AI识别结果：

模型输出：
{clean_model_output(row["model_output"])}

模型判断：
{result}

检测框：
{boxes}


最终审核结果：
{"违法" if str(row['is_committed'])=="1" else "未确认违法"}


图片地址：
{row['图片路径']}

"""


    markdown_list.append(md)



    # --------------------------
    # Metadata
    # --------------------------

    metadata = {

        "case_id": str(row["id"]),

        "illegal_type":
            str(row["违法类型"]),

        "location":
            str(row["发生地点"]),

        "time":
            str(row["发生时间"]),

        "team":
            str(row["所属支队"]),

        "image_url":
            str(row["图片路径"]),

        "model_result":
            result,

        "bounding_boxes":
            boxes,

        "is_committed":
            int(row["is_committed"])
            if pd.notna(row["is_committed"])
            else None
    }


    metadata_list.append(metadata)



# ==============================
# 输出Markdown
# ==============================

# 每个 MD 文件保存多少条案例
cases_per_file = 300

# 输出目录
output_dir = "违法案例库"
os.makedirs(output_dir, exist_ok=True)

# 按 300 条进行切分
for start in range(0, len(markdown_list), cases_per_file):

    # 当前文件的案例
    current_cases = markdown_list[
        start:start + cases_per_file
    ]

    # 文件编号
    file_index = start // cases_per_file + 1

    # 文件路径
    md_path = os.path.join(
        output_dir,
        f"违法案例_{file_index:03d}.md"
    )

    # 拼接案例
    content = "\n========================\n".join(
        current_cases
    )

    # 写入文件
    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content)

    print(f"已生成：{md_path}，案例数：{len(current_cases)}")



# ==============================
# 输出Metadata JSON
# ==============================

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata_list,
        f,
        ensure_ascii=False,
        indent=4
    )


print("生成完成")
print(md_path)
print(metadata_path)