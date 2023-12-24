# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:26:02 2023

@author: ja983
"""
import cv2
import pytesseract
from PIL import Image
import re
image = cv2.imread('TEST1123.jpg')
 #圖像預先處理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

pil_image = Image.fromarray(binary)
text = pytesseract.image_to_string(Image.fromarray(binary), lang='eng+chi_sim', config="--psm 6")

# 假设你的文字是以下形式
#text = " 中一 一 二 一[io no   OFC FULL DaY FARAKEA_RANK FARAREAFEE_FLS FARAREAFEE_DIS_FLS ]中 17354 al 24108    LNULL     NULL      NULL       |2300561 B222222222 24000    LNULL 。   NUILL      NULL      -|."
text2 = pytesseract.image_to_string(image, lang='eng', config="--psm 6 --oem 3")
print(f'二值化後的資料:{text}')
print(f'直接將圖像文字轉字串:{text2}')
# 将NumPy数组转换为PIL图像

#text = "：200561 B222222222 240001"
# 定義身分證字號規則
id_number_pattern = re.compile(r'[A-Za-z]\d{9}')

# 比對是否有符合規則的字串
matches = id_number_pattern.findall(text)
matches2 = id_number_pattern.findall(text2)

# 输出比對到的身份证字号
for match in matches:
    print(f"二值化後資料比對到哦字串: {match}")
    
for match in matches2:
    print(f"直接將圖像文字轉字串後比對到的字串: {match}")
    