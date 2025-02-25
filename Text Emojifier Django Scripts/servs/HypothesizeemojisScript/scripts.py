from keras.models import model_from_json
import pandas as shifu
import numpy as nu
import emoji

DICTEMOJIS = {"0": "\u2764\uFE0F",
              "1": ":baseball:",
              "2": ":smiling_face_with_smiling_eyes:",
              "3": ":disappointed_face:",
              "4": ":fork_and_knife:",
              "5": ":hundred_points:",
              "6": ":fire:",
              "7": ":face_blowing_a_kiss:",
              "8": ":chestnut:",
              "9": ":flexed_biceps:"
             }

with open("servs\HypothesizeemojisScript\hypothesizeemojisarch.json", "r") as f:
    threelaparch = model_from_json(f.read())
threelaparch.load_weights("servs\HypothesizeemojisScript\hypothesizeemojisarch.weights.h5")
threelaparch.make_predict_function()
fgi = open("servs\HypothesizeemojisScript\glove.6B.50d.txt", encoding = "utf-8")

GLOVE_DICT = {}
cnts = 0
for txts in fgi:
    attrs = txts.split()
    txt = attrs[0]
    vctr = nu.asarray(attrs[1:], dtype = "float")
    GLOVE_DICT[txt] = vctr
fgi.close()

dimemvctrstxts = GLOVE_DICT["cat"].shape[0]

def coutemmatrix(trst):
    maxlength = 30
    vctremmatrixtxt = nu.zeros((trst.shape[0],maxlength, dimemvctrstxts))
    for j in range(trst.shape[0]):
        trst[j] = trst[j].split()
        for i in range(len(trst[j])):
            try:
                vctremmatrixtxt[j][i] = GLOVE_DICT[trst[j][i].lower()]
            except:
                vctremmatrixtxt[j][i] = nu.zeros((50,))
    return vctremmatrixtxt

def hypothesis(iout):
    ix = shifu.Series([iout])
    hematrix = coutemmatrix(ix)

    h = threelaparch.predict(hematrix)
    classes_x=nu.argmax(h,axis=1)

    return emoji.emojize(DICTEMOJIS[str(classes_x[0])])

if __name__ == "__main__":
    print(hypothesis("Hey man how are you doing?"))