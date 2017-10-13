import math
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from random import *

n = 2
w= 60
h= 60
noise =0.1
gen= 20000
fonts = [
    ImageFont.truetype("lucida sans italic.ttf", 25),
    ImageFont.truetype("Lucida Sans Bold.ttf", 25),
    ImageFont.truetype("lucidasansregular.ttf", 25),
]

chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
 'x', 'y', 'z']

text_colors = [
    (0,255,255),
    (0,0,0),
    (255,255,255)
]

background_colors = [
   (255,255,255),
    (0,0,0,0)
]

gradient_colors = [
    (255,0,0),
    (0,0,255),
    (255,255,0), 
]

def randof(arr):
    return arr[randrange(len(arr))]

def gradient(img, w, h, innerColor, outerColor, noise=0.0):
    x_center = randrange(w) +1
    y_center = randrange(h)
    for y in range(h):
        for x in range(w):
            rnd = random()
            if rnd>noise:
                #Find the distance to the center
                distanceToCenter =  math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) / 10

                #Make it on a scale from 0 to 1
                distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * x_center)

                #Calculate r, g, and b values
                r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
                g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
                b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)

                #Place the pixel        
                img.putpixel((x, y), (int(r), int(g), int(b)))
import csv
import progressbar
bar = progressbar.ProgressBar()
with open('data/captcha_labels.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path','inputted_captcha'])

    for x in bar(range(gen)):
        font = randof(fonts)
        random_chars = [randof(chars) for _i in range(n)]
        label= ''.join(random_chars)
        written_str = ' '.join(random_chars)

        text_color=randof(text_colors)
        background_color = randof(background_colors)

        img = Image.new("RGB", (w,h), background_color)
        draw = ImageDraw.Draw(img)

        inner_color = randof(gradient_colors)
        tmp = gradient_colors[:]
        tmp.remove(inner_color)
        outer_color = randof(tmp)

        gradient(img, w, h, inner_color, outer_color, noise=noise)

        y_pos = randrange(h - 35)
        draw.text((0, y_pos),written_str,text_color,font=font)

        path='data/sample-out_'+str(x)+'.jpg'
        img.save(path)
        writer.writerow([path,label])
