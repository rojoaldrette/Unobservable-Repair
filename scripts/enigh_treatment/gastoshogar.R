# _____________________________________________________________________________
#
# Proyecto:       Tesis_final_1
#
# Script:         scripts/enigh_treatment/gastoshogar.R
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
#cat("\014")           # Limpiar consola


# Carga de paquetes
pacman::p_load(tidyverse, readxl, writexl, janitor, lubridate, jsonlite, here)
  
# CODIGO ______________________________________________________________________

# FUNCIONES -------------------------------------------------------------------


options()


get_codes <- function(year, df){
  df <- df %>% filter(year == year)
  return(df$code)
}


filter_treatment <- function(df, year, df2, factor=NA){


  df <- df %>% filter(clave %in% get_codes(year, df2))
  df <- df %>% mutate(gasto_total = gasto * factor) %>% 
    filter((gasto_total != 0) | (!is.na(gasto_total))) %>%
    group_by(clave) %>% summarise(gasto_total = sum(gasto_total)) %>%
    mutate(gasto_total = gasto_total / 1000000)

  return(df)
}




# DATA ------------------------------------------------------------------------

# Códigos por año
codes <- readxl::read_xlsx("input/cosos/filtros_enigh.xlsx")


# Gastoshogar 24 ######################################################
gh24 <- read.csv("input/enigh/gastoshogar24.csv")
gh24_1 <- filter_treatment(gh24, 2024, codes)

# Gastoshogar 22 ######################################################
gh22 <- read.csv("input/enigh/gastoshogar22.csv")
gh22_1 <- filter_treatment(gh22, 2022, codes)


# Gastoshogar 20 ######################################################
viv20 <- read.csv("input/enigh/viviendas20.csv") %>% select(folioviv, factor)
gh20 <- read.csv("input/enigh/gastoshogar20.csv")
gh20_d <- left_join(gh20, viv20, by="folioviv")
gh20_d <- gh20_d %>% group_by(folioviv, foliohog) %>% mutate(repeats = n())

gh20_1 <- filter_treatment(gh20, 2020, codes)


# Gastoshogar 18 ######################################################
gh18 <- read.csv("input/enigh/gastoshogar18.csv")
gh18_1 <- filter_treatment(gh18, 2018, codes)



# Fucking enigh
# Issues: the folioviv appears like shit with notation
# I have to put factor from viviendas into 2020 and below, this sucks so much wth




# SCRIPT ----------------------------------------------------------------------