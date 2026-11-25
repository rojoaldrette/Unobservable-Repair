# Diagnóstico: E[w | j, a] y tasa de avería verdadera vs. la que implica el logit de Gillingham.
# Usa el equilibrio verdadero ya resuelto (no resuelve nada).  python -u claude/graphs/diag_w_por_edad.py
import numpy as np
import pandas as pd
import hacer_graficas as H

import sys
PESO = sys.argv[1] if len(sys.argv) > 1 else "hold"   # hold = tenencias (a las que aplica s); q = stock
cfg, reg = H.objetos_verdaderos()
g = cfg.w_grid
S = pd.read_csv(H.TAB / "resumen_parametros.csv").set_index(["estimador", "parametro"])
gi = lambda p: S.loc[("gill_completa", p), "media"]
filas = []
for t, tipos in enumerate(reg):
    for j in range(2):
        for a in [1, 3, 5, 8, 12, 16, 20, 23]:
            # stock al inicio del año (q), juntando tipos
            qs = sum(tp["f"] * np.asarray(tp[PESO])[j, a - 1] for tp in tipos)
            s_true = sum(tp["f"] * np.asarray(tp[PESO])[j, a - 1] * np.asarray(tp["s"])[j, a]   # s se indexa por edad (0 = nuevo)
                         for tp in tipos)
            m = qs.sum()
            ew = (qs * g).sum() / m
            sdw = np.sqrt((qs * (g - ew) ** 2).sum() / m)
            jj = f"j{j}"
            s_g = 1 / (1 + np.exp(-(gi(f"acc_int_{jj}") + gi(f"acc_age_{jj}") * a)))
            filas.append(dict(reg=H.REG_MINI[t], marca=H.MARCAS_COCHE[j], a=a, Ew=ew, DEw=sdw,
                              s_verdad=s_true.sum() / m, s_gill=s_g))
pd.set_option("display.width", 200)
print(pd.DataFrame(filas).round(4).to_string(index=False))
