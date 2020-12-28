
import pickle
import sklearn.metrics as rocUtils
from predict_model import PredictModel

import os
basedir = os.path.dirname(__file__)
import matplotlib.pyplot as plt


pModel = PredictModel()

print(basedir)
def perdictDisease(s):
    rs = pModel.predict(s)
    return rs



if __name__ == '__main__':

    s1 = "1.腰椎骨折(L1,爆裂性) 2.胸椎退行性病变 3.腰椎退行性病变 4.S2椎体血管瘤。入院情况:患者以高处坠落致腰背部肿痛、活动受限2天。”为主诉入院,专科情况:车送入院。腰椎生理曲度正常,无明显侧弯畸形。腰椎活动受限。腰1椎体水平明显压痛及叩击痛,无放射痛。双下肢感觉正常,双髂腰肌、股四头肌、胫前肌、腘绳肌、踇长伸肌5级,双足背伸、跖屈肌力5级。双直腿抬高试验、股神经牵拉试验阴性。双膝、踝反射正常。双巴彬斯基征阴性。双上肢未见明显异常体征。主要治疗经过:入院后予完善相关检查,于2019-11-28在全麻+局麻下行腰1椎体成形术,术后给予预防感染、止痛、促进愈合、改善微循环、补液等处理。术后恢复良好,要求出院,予办理。出院时情况:患者诉切口稍疼痛,无其他特殊不适。查体：神清,生命征平稳,心肺腹无明显异常。切口敷料干燥,无渗血、渗液。双下肢感觉正常,双足背无肿胀,双足趾、踝关节活动良好,双足背动脉搏动可触及,末梢血运好。1、骨科门诊随诊,每周复查1次。2、切口门诊换药,2-3天一次。3、继续卧床休息,腰部避免负重,加强腰背肌及四肢功能锻炼。4、建议休息叁个月。不适随诊。"
    s2 = "1.腰椎骨折(L1,爆裂性)2.胸椎退行性病变3.腰椎退行性病变4.S2椎体血管瘤"
    s3 = "管瘤"

    sArr = []
    sArr.append(s1)
    sArr.append(s2)
    sArr.append(s3)

    value,scores = perdictDisease(sArr)
    for v in value:
        print(v)
    for v in scores:
        print(v)