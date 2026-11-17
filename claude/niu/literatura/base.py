"""Constructores de entradas del catálogo. T = lo tienes en po/; F = falta."""


def T(clave, resumen, archivo="", escaneado=False):
    return {"clave": clave, "resumen": resumen, "archivo": archivo, "escaneado": escaneado}


def F(clave, resumen, prioridad=False):
    return {"clave": clave, "resumen": resumen, "prioridad": prioridad}
