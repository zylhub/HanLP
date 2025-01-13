import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH)
doc = HanLP(['晋太元中，武陵人捕鱼为业。', '司馬牛問君子'])
print(doc)
doc.pretty_print()

HanLP('司馬牛問君子', skip_tasks='tok/fine').pretty_print()
