import json, glob

import json, glob, collections
shapes = collections.Counter()
paths = glob.glob("data/precedents/**/*.json", recursive=True)
for p in paths:
    try:
        with open(p, encoding="utf-8") as f:
            js = json.load(f)
    except Exception:
        shapes["invalid_json"] += 1
        continue
    if isinstance(js, dict):
        if "PrecService" in js: shapes["dict_PrecService"] += 1
        elif any(k in js for k in ["판례목록","precList","prec_list"]): shapes["dict_list_wrapper"] += 1
        elif any(k in js for k in ["판례일련번호","판결요지","판시사항","사건명","법원명"]): shapes["dict_flat"] += 1
        else: shapes["dict_other"] += 1
    elif isinstance(js, list):
        shapes["list_top"] += 1
    else:
        shapes["other_top"] += 1

print(shapes)
