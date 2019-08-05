# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from PIL import Image

classes = ["n02123159","n02676566","n02773838","n03179701",
           "n03255030","n03445777","n03642806","n03792782",
           "n04037443","n04555897"];

for c in classes:
           
    path = c + '/Annotations/'          
    xmls = os.listdir(path)
    
    exportFolderPath = "cropped/" +  c + '/'
    os.makedirs(exportFolderPath);
    
    for x in xmls:
        try:
#            
            print("Cropping: " + x)
            xmlPath = path + x            
            file = open(xmlPath)
            root = ET.fromstring(file.read())
            file.close()
            
            xmin = int (root.find('object').find('bndbox').find('xmin').text)
            ymin = int (root.find('object').find('bndbox').find('ymin').text)
            xmax = int (root.find('object').find('bndbox').find('xmax').text)
            ymax = int (root.find('object').find('bndbox').find('ymax').text)
            
            imgName = os.path.splitext(x)[0] + '.JPEG'
            imgPath = c + '/' + imgName
            
            im = Image.open(imgPath)
            croppedIm = im.crop((xmin, ymin, xmax, ymax))
            
            width, height = croppedIm.size
            
            if width > 300 and height > 300:
                
                if width > height:
                    
                    new_height = int((300/width)*height)
                    croppedIm = croppedIm.resize((300, new_height), Image.ANTIALIAS)
                     
                else:
                    
                    new_width = int((300/height)*width)
                    croppedIm = croppedIm.resize((new_width, 300), Image.ANTIALIAS)
      
                    
            elif width <= 300 and height > 300:
                
                new_width = int((300/height)*width)
                
                croppedIm = croppedIm.resize((new_width, 300), Image.ANTIALIAS)

            elif width > 300 and height <= 300:
                
                new_height = int((300/width)*height)
                
                croppedIm = croppedIm.resize((300, new_height), Image.ANTIALIAS)

            
            im.close()
            
            #Save the cropped image
            croppedImPath = "cropped/" + imgPath
            croppedIm.save(croppedImPath, 'JPEG')
            
            
        except:
            
            print("An exception occurred for " + x)
          
