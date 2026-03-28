import json
d = json.load(open('final_eval.json', encoding='utf-8'))
for ds, atypes in d.items():
    for at, v in atypes.items():
        print(ds, at, 'gold='+str(v['gold_n']), 'tp='+str(v['tp']), 'fp='+str(v['fp']), 'fn='+str(v['fn']), 'P='+str(v['P']), 'F1='+str(v['F1']))
