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
pacman::p_load(tidyverse, readxl, writexl, janitor, lubridate, jsonlite,
          ggplot2)
  
# CODIGO ______________________________________________________________________

# FUNCIONES -------------------------------------------------------------------


options(scipen = 999)

get_codes <- function(year, df){
  df <- df %>% filter(year == year)
  return(df$code)
}


filter_treatment <- function(df, year, df2){


  df <- df %>% filter(clave %in% get_codes(year, df2))
  df <- df %>% mutate(gasto_total = gasto * factor) %>% 
    filter((gasto_total != 0) | (!is.na(gasto_total))) %>%
    group_by(clave) %>% summarise(gasto_total = sum(gasto_total)) %>%
    mutate(gasto_total = gasto_total / 1000000) %>% ungroup() %>%
    rename(code = clave) %>% mutate(year = year)

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
hog20 <- read.csv("input/enigh/hogares20.csv") %>% select(folioviv, foliohog)
gh20 <- read.csv("input/enigh/gastoshogar20.csv")

hog20_2 <- hog20 %>%
  left_join(
    viv20,
    by = "folioviv"
  ) %>%
  group_by(folioviv) %>%
  mutate(
    n_hog = n(),
    factor = factor / n_hog
  )

gh20_d <- gh20 %>% left_join(hog20_2, by=c("folioviv", "foliohog"))
gh20_1 <- filter_treatment(gh20_d, 2020, codes)


# Gastoshogar 18 ######################################################

viv18 <- read.csv("input/enigh/viviendas18.csv") %>% select(folioviv, factor)
hog18 <- read.csv("input/enigh/hogares18.csv") %>% select(folioviv, foliohog)
gh18 <- read.csv("input/enigh/gastoshogar18.csv")

hog18_1 <- hog18 %>%
  left_join(
    viv18,
    by = "folioviv"
  ) %>%
  group_by(folioviv) %>%
  mutate(
    n_hog = n(),
    factor = factor / n_hog
  )

gh18_d <- gh18 %>% left_join(hog18_1, by=c("folioviv", "foliohog"))
gh18_1 <- filter_treatment(gh18_d, 2018, codes)



# SCRIPT ----------------------------------------------------------------------


df_comb <- bind_rows(gh24_1, gh22_1, gh20_1, gh18_1) %>%
  group_by(year) %>%
  left_join(codes, by=c("year", "code")) %>% filter(keep == 1) %>%
  select(year, gasto_total, label2)

save()


p <- ggplot(df_comb, aes(x=year, y=log(gasto_total), color=label2, group=label2))+
      geom_line(size=2)+
      geom_point(size=3)+
      theme_bw()+
      labs(title="Gasto en mantenmiento y reparación",
          x="Año", y="log(mdp)", color="Tipo de gasto")+
      theme(text = element_text(size = 18)) 

ggsave("output/graphs/gasto_rep.png", plot=p, height = 6, width=8)





