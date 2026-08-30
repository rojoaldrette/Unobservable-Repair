# _____________________________________________________________________________
#
# Proyecto:       Tesis_final_1
#
# Script:         scripts/funciones.R
# Objetivo:       Funciones auxiliares del proyecto
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          30/08/2026
#
# _____________________________________________________________________________

# FUNCIONES GENERALES _________________________________________________________

#' [Título] Ejemplo de función documentada
#' 
#' @descripcion Función de ejemplo que demuestra la estructura recomendada
#' @param x Vector numérico de entrada
#' @return Vector numérico procesado
#' @ejemplo
#'   resultado <- funcion_ejemplo(c(1, 2, 3))
#' @export
funcion_ejemplo <- function(x) {}

#' Carga de variables de entorno desde .env
#' @descripcion Carga las variables de entorno definidas en un archivo .env
#' ubicado en el directorio raíz del proyecto. Esta es una buena práctica para
#' manejar configuraciones sensibles como claves API, rutas de archivos, etc.
pacman::p_load(dotenv); load_dot_env(file = ".env")

# FUNCIONES DE CARGA DE DATOS _________________________________________________

# Funciones de carga del catálogo de localidades/municipios/estados del INEGI
# Importa 2 funciones, una para crear llaves de matching, llamada
# crear_llave_matching() y otra para traer el catálogo de INEGI, llamada
# get_geo_inegi()
source("scripts/utils/funciones_cat_geo_inegi.R")

# FUNCIONES DE PROCESAMIENTO __________________________________________________



# FUNCIONES DE VISUALIZACIÓN __________________________________________________



# FUNCIONES DE EXPORTACIÓN ____________________________________________________


