import cv2
import pytesseract
import numpy as np
import imutils
import re
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Soft\Tesseract\tesseract.exe'

def image_to_text(image):
    
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, image = cv2.threshold(image, 127,255, cv2.THRESH_BINARY)
    # edged = cv2.Canny(image, 50, 200, 255)
    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,5)
    # image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    image = ~image
    
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,4)
    image = ~image
    kernel = np.ones([3, 3])
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = ~opening
    
    
    
    text = pytesseract.image_to_string(image, config='--psm 6')
    cv2.imshow(text, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return text
 
def get_top(image):
    h, w = image.shape[:-1]
    return image[0:80, w//5: w-w//5]

def get_gold(image):
    image = get_top(image)
    coin = cv2.imread('./coin.jpg')
    w, h = coin.shape[:-1]

    # find coin to extract gold near of it
    res = cv2.matchTemplate(image, coin, cv2.TM_CCOEFF_NORMED)
    

    loc = np.where(res >= 0.8)

    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)

    # cv2.imshow('', image)
    # cv2.waitKey()

    
    top_h, top_w = image.shape[:-1]

    # find the location of teams' gold
    blue_loc = loc[1][np.where(loc[1] < top_w //2 - 10)]
    red_loc = loc[1][np.where(loc[1] > top_w // 2 + 10)]
    blues_coin_x = blue_loc[0]
    reds_coin_x = red_loc[0]

    # crop teams' gold
    blue_gold = image[h-15: h + 15, blues_coin_x + w+5: blues_coin_x+w+60]
    red_gold = image[h-15: h + 15, reds_coin_x + w + 5: reds_coin_x+w+60]
    
    # converting itself
    blue_num_gold = image_to_text(blue_gold)
    red_num_gold = image_to_text(red_gold)
    
    # comass to periods
    blue_num_gold = blue_num_gold.replace(',', '.')
    red_num_gold = red_num_gold.replace(',', '.')

    # remove spaces
    blue_num_gold = blue_num_gold.strip()
    red_num_gold = red_num_gold.strip()

    # remove k 
    if blue_num_gold[-1] == 'k':
        blue_num_gold = blue_num_gold[:-1]
    if red_num_gold[-1] == 'k':
        red_num_gold = red_num_gold[:-1]
    
    try:
        blue_num_gold = float(blue_num_gold)
    except:
        print('gold problem')
        plt.imshow(blue_gold)
        plt.show()
        blue_num_gold = float(input('input blue gold'))

    try:
        red_num_gold = float(red_num_gold)
    except:
        print('gold problem')
        plt.imshow(red_gold)
        plt.show
        red_num_gold = float(input('input red gold'))
    
    if blue_num_gold < 1000.0:
        blue_num_gold *= 1000
        red_num_gold *= 1000

    return(blue_num_gold, float(red_num_gold))

def get_score(image):
    h, w = image.shape[:-1]
    # extract scores itself
    blue_score = image[h - h//4 + 25: h, w // 2 - 120: w//2 -70]
    red_score = image[h - h//4 +25: h, w // 2 + 80: w // 2 + 125]
    blue_val = image_to_text(blue_score)
    red_val = image_to_text(red_score)

    blue_val = blue_val.replace('O', '0')
    blue_val = blue_val.replace('o', '0')

    red_val = red_val.replace('O', '0')
    red_val = red_val.replace('o', '0')

    reds = []
    blues = []

    # I gonna find pattern the pattern
    pattern = '\d+/\d+/\d+'

    blues = re.findall(pattern, blue_val)
    reds = re.findall(pattern, red_val)
    
    print(blues, reds)

    for i in range(len(blues), 5):
        print('blues collected {0}'.format(blues))
        plt.imshow(blue_score)
        plt.show()

        blues.append(input('missing:'))
    
    for i in range(len(reds), 5):
        print('reds collected {0}'.format(reds))
        plt.imshow(red_score)
        plt.show()
        reds.append(input('missing:'))

    blue_kills = 0
    red_kills = 0
    blue_assists = 0
    red_assists = 0

    # simple spliting and get all needed values
    for red in reds:
        splited = red.split('/')
        red_kills += int(splited[0])
        red_assists += int(splited[2])
    
    for blue in blues:
        splited = blue.split('/')
        blue_kills += int(splited[0])
        blue_assists += int(splited[2])
    
    return (blue_kills, blue_assists, red_kills, red_assists)



def get_total_minions(image):
    h, w = image.shape[:-1]

    blue_minions = image[h - h//4 + 25: h, w // 2 - 70: w//2 -30]
    red_minions = image[h - h//4 +25: h, w // 2 + 40: w // 2 + 70]

    blue_val = image_to_text(blue_minions)
    
    blue_val = blue_val.split('\n')
    
    red_val = image_to_text(red_minions)
    red_val = red_val.split('\n')

    for i in range(len(blue_val), 5):
        print('blue minions missing {0}'.format(blue_val))
        plt.imshow(blue_minions)
        plt.show()
        blue_val.append(input('missing minion:'))

    for i in range(len(red_val), 5):
        print('red minions missing {0}'.format(red_val))
        plt.imshow(red_minions)
        plt.show()
        red_val.append(input('missing minion:'))

    blue_total = 0

    i = 0
    for blue in blue_val:
        try:
            blue = int(blue)
        except:
            print('not integer blue {0}'.format(i))
            plt.imshow(blue_minions)
            plt.show()
            blue = int(input('correct val:'))
        blue_total += blue
        i += 1
    i = 0

    red_total = 0
    for red in red_val:
        try:
            red = int(red)
        except:
            print('not integer red {0}'.format(i))
            plt.imshow(red_minions)
            plt.show()
            red = int(input('correct val:'))
        red_total += red
        i += 1
        

    return (blue_total, red_total)
        
def get_minions(image):
    
    # it didnt work well with low res digits
    # so a litle bit of preproc
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = ~img
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,4)
    img = ~img
    kernel = np.ones([3, 3])
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = opening
    
    # find contours for detecting gap betwen digits
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    
    # at first sort it by area, then by position(from top to bot) 
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])[:7]
    
    
    first = cv2.boundingRect(cnts[0])[1]
    cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
    cv2.imshow('',image)
    cv2.waitKey()
    # find gap betwen 2 first digits
    gap = 0
    for cnt in cnts:
        new_y = cv2.boundingRect(cnt)[1]
        
        if new_y > first + 40:
            gap = new_y - first
            break
    
    # when gap is found I make array of digit position
    edges = np.arange(first , first + 5*gap, gap)
    total = 0
    
    # count total minions 
    for gy in edges:
        new_img = cv2.resize(image[gy-5: gy + 40, :], None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        val = image_to_text(new_img)
        try:
            total += int(val)
        except :
            
            print('wtf {0}'.format(val))
            plt.imshow( new_img)
            plt.show()
            
            total += int(input('input correct val:'))

    return (total, edges)

        
def get_time(image):
    top = get_top(image)
    top_h, top_w = top.shape[:-1]
    time = top[top_h//2: top_h, top_w//2 - 20: top_w//2 + 20]
    text = image_to_text(time)
    return text



img = cv2.imread('./1.png')

blue_minions, red_minions = get_total_minions(img)

blue_kills, blue_assists, blue_death, red_assists = get_score(img)

print('gold')
blue_gold, red_gold = get_gold(img)
print(blue_gold, red_gold)
print('minions {0}, {1}'.format(blue_minions, red_minions))
print('bk {0}, bass {1}, bdeath {2}, red_ass {3}'.format(blue_kills, blue_assists, blue_death, red_assists))
# print(time)
cv2.destroyAllWindows()
variables = [blue_gold, red_gold, blue_minions, red_minions, blue_kills, blue_death, blue_assists, red_assists]
variables = [str(i) for i in variables]
print(variables)
with open('vars.txt', 'w') as f:
    a = " "
    a = a.join(variables)
    print(a)
    f.write(a)


