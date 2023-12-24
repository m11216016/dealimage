import cv2
import pytesseract
from PIL import Image
import re

# 读取图像
#image = cv2.imread('TEST1123.jpg')
my_list = [1, 2, 3, 4, 5]
my_list[1] = "ID12132.jpg"
my_list[0] = "TEST1123.jpg"
my_list[2] = "ID1213.jpg"
my_list[3] = "ID12131.jpg"
my_list[4] = "id5.jpg"
# 使用 for 迴圈讀取元素
for element in my_list:
    image = cv2.imread(element)
    cv2.imshow('origion Image', image)
    cv2.waitKey(0)
    # 使用 pytesseract 將圖像文字轉字串
    text_data = pytesseract.image_to_data(Image.fromarray(image), output_type=pytesseract.Output.DICT)
    
    # 打印 text_data 
    #print(text_data)
    
    # 显示文字區域數浪
    #print("Number of text regions:", len(text_data['text']))
    id_number_pattern = re.compile(r'[A-Za-z]\d{9}')
    text2 = pytesseract.image_to_string(image, lang='eng', config="--psm 6 --oem 3")
    # 比對是否有符合規則的字串
     
    #matches2 = id_number_pattern.findall(text2)
    
    # 输出比對到的身份证字号
    #for match in matches2:
    #    print(f"二值化後資料比對到哦字串: {match}")
        
    #for match in matches2:
    #    print(f"直接將圖像文字轉字串後比對到的字串: {match}")
    
    for i in range(len(text_data['text'])):
           #print (text_data['conf'][i])
           if int(text_data['conf'][i]) > 0:
                # 擷取文中信息和位置
                text = text_data['text'][i]
                #print(i);
                #print(text);
                matches2 = id_number_pattern.findall(text)
    
                # 输出比對到的身份证字号
                #for match in matches2:
                #    print(f"二值化後資料比對到哦字串: {match}")
                
                for match in matches2:
                 
                    x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                    #print(text,x,y,w,h)
                    # 添加條件檢查，確保 ROI 在圖像内
                    #print(x)
                    #if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and 0 <= w < image.shape[1] and 0 <= h < image.shape[0]:
                        # 針對所找到的區域做模糊處理
                    roi = image[y:y+h, x:x+w]
                    #print(h);
                    #print(roi.size);
                    ##roi = image[y:h+30, x:w+30, :] = blurred_roi[:, :w+30-x, :]
                    if not roi.size == 0:  #  
                     
                        #cv2.waitKey(0)
                    #if not x == -1:
                        blurred_roi = cv2.blur(roi, (100, 100), 0)
                        print(f"直接將圖像文字轉字串後比對到的字串: {match}")
                            # 将模糊處理後的區域放回原圖
                        #print(x)
                        image[y:y+h, x:x+w] = blurred_roi
    
            # 顯示结果
    cv2.imshow('Blurred Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()