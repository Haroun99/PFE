import re
filename= '/home/haroun/Vitis-AI/Vitis-AI-Tutorials-2.5/Tutorials/Keras_FCN8_UNET_segmentation/files/code/requirements.txt'
new_txt=''
with open(filename, 'r') as f :
    txt=f.readlines()
    for line in txt:
        new_line = re.sub(' +','==',line)
        new_txt=new_txt+new_line
with open(filename, 'w') as f :
    f.write(new_txt)