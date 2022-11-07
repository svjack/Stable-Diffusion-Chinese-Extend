from init_model import *

path = "svjack/prompt-extend-chinese"
#### upload to huggingface
mt5_b_cpu = T5_B(path,
    device = "cpu")

if __name__ == "__main__":
    mt5_b_cpu.predict_batch(
        [
            "卡通龙",
            "第一次世界大战",
            "维多利亚女王"
        ]
    )
    '''
    ['数字艺术、艺术品趋势、电影照明、工作室质量、光滑成型、5',
     '在艺术站的潮流,8,高度详细,高质量,高分辨率,获',
     '由,和,制作的艺术作品,在站上趋势,8,超宽']
    '''
