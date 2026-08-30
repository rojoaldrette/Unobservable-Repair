# _____________________________________________________________________________
#
# Proyecto:       Tesis_final_1
#
# Script:         scripts/main.R
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          30/08/2026
# 
# Última
# actualización:  30/08/2026
#
# _____________________________________________________________________________

# PREAMBULO ___________________________________________________________________

# Limpiar entorno de trabajo
rm(list = ls())       # Limpiar entorno de trabajo
source("~/.Rprofile") # Cargar configuraciones globales
cat("\014")           # Limpiar consola


# Carga de paquetes
pacman::p_load(tidyverse, readxl, writexl, janitor, lubridate, jsonlite)
  
# CODIGO ______________________________________________________________________

# FUNCIONES -------------------------------------------------------------------
source("./scripts/funciones.R")

# DATA ------------------------------------------------------------------------

# SCRIPT ----------------------------------------------------------------------

