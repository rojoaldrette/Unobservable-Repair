# Diagnóstico de la figura 4: precio de usados por edad, verdad vs. diseños vs. Gillingham (régimen base).
# python -u claude/graphs/diag_precios.py
import pandas as pd
import hacer_graficas as H

mc = H.FINAL / "output/modelo_tesis/montecarlo/base"
pe, ge = H.cargar_por_edad(mc)
print("columnas por_edad:", list(pe.columns))
print("columnas gill:", list(ge.columns))
b = pe[pe.regimen == 1]
v = b[b.estimador == "verdad"].set_index(["j", "a"]).P_transaccion.rename("verdad")
t = pd.concat([v] + [b[b.estimador == e].groupby(["j", "a"]).P_transaccion.mean().rename(e)
                     for e in H.ESTS[1:]]
              + [ge.groupby(["j", "a"]).P.mean().rename("gill")], axis=1)
t["gill_%"] = 100 * (t.gill / t.verdad - 1)
t["d3_%"] = 100 * (t.diseno_3 / t.verdad - 1)
# verdad en los otros regímenes (para ver cuánto mueve R el precio)
for r in (0, 2):
    t[f"verdad_reg{r}"] = pe[(pe.regimen == r) & (pe.estimador == "verdad")].set_index(["j", "a"]).P_transaccion
pd.set_option("display.width", 220)
print(t.loc[[(j, a) for j in (0, 1) for a in (1, 2, 3, 5, 8, 12, 16, 20, 23, 24) if (j, a) in t.index]]
      .round(2).to_string())
