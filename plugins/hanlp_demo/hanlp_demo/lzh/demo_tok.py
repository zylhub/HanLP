import hanlp

HanLP = hanlp.load(hanlp.pretrained.tok.KYOTO_EVAHAN_TOK_LZH)
doc = HanLP('司馬牛問君子')
print(doc)
