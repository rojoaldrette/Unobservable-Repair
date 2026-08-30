# _____________________________________________________________________________
#
# Proyecto:       Tesis_final_1
#
# Script:         scripts/funciones_cat_geo_inegi.R_1.r
# Objetivo:       Funciones para trabajar con el catálogo geográfico de INEGI.
#                 Incluye: crear_llave_matching() y get_geo_inegi(). Requiere
#                 acceso a la base de datos PostgreSQL vía VPN.
#
# Autor:          Jorge A. Gómez
# Correo(s):      jorge.gomez@transformaciondigital.gob.mx
#
# Fecha:          29/10/2025
# 
# Última
# actualización:  01/11/2025
#
# _____________________________________________________________________________



# FUNCIONES DE USO DEL CATÁLOGO DEL INEGI ______________________________________

#' Crear llaves de matching en un data.frame/tibble
#'
#' Crea columnas de llave de matching para entidad+municipio y, opcionalmente,
#' entidad+municipio+localidad. La estandarización de nombres sigue la misma
#' lógica utilizada en `procesar_claves_combinadas`: title case respetando
#' `stopwords_title_es`, mantener números romanos en mayúsculas y luego
#' transliterar a snake_case (Latin-ASCII). El parámetro `nombres_ent_cortos`
#' (por defecto TRUE) solo afecta la versión usada para construir la llave
#' y no modifica las columnas originales del data.frame.
#'
#' @param df Data frame o tibble con los datos.
#' @param ent Nombre de la columna con el nombre de la entidad (string).
#' @param mun Nombre de la columna con el nombre del municipio (string).
#' @param loc Nombre de la columna con el nombre de la localidad (string). Opcional, por defecto NULL.
#' @param nombres_ent_cortos Logical. Si TRUE, aplica simplificación (ej. "Coahuila de Zaragoza" → "Coahuila")
#'        solo en la versión usada para crear las llaves de nombre. No modifica las columnas originales.
#' @param sobreescribir_resultados Logical. Si FALSE, evita sobrescribir columnas ya existentes de llaves.
#'
#' @return Data frame/tibble con las nuevas columnas añadidas.
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   NOM_ENT = c("COAHUILA DE ZARAGOZA", "VERACRUZ DE IGNACIO DE LA LLAVE", "MICHOACÁN DE OCAMPO", 
#'               "JALISCO", "NUEVO LEÓN", "GUANAJUATO", "PUEBLA", "CHIAPAS", "CIUDAD DE MÉXICO", "OAXACA"),
#'   NOM_MUN = c("SALTILLO", "XALAPA", "MORELIA", "GUADALAJARA", "MONTERREY", 
#'               "LEÓN", "PUEBLA", "TUXTLA GUTIÉRREZ", "BENITO JUÁREZ", "OAXACA DE JUÁREZ"),
#'   NOM_LOC = c("SALTILLO", "XALAPA-ENRÍQUEZ", "MORELIA", "GUADALAJARA", "MONTERREY",
#'               "LEÓN DE LOS ALDAMA", "PUEBLA DE ZARAGOZA", "TUXTLA GUTIÉRREZ", 
#'               "CIUDAD DE MÉXICO", "OAXACA DE JUÁREZ")
#' )
#' df2 <- crear_llave_matching(df, ent = "NOM_ENT", mun = "NOM_MUN")
#' df3 <- crear_llave_matching(df, ent = "NOM_ENT", mun = "NOM_MUN", loc = "NOM_LOC")
#' }
#' @export
crear_llave_matching <- function(
  df,
  ent = "NOM_ENT",
  mun = "NOM_MUN",
  loc = NULL,
  nombres_ent_cortos = TRUE,
  sobreescribir_resultados = TRUE
) {
  
  # Paquetes mínimos necesarios
  pacman::p_load(dplyr, stringr, snakecase, crayon)

  #' Palabras que deben quedar en minúsculas en títulos en español
  stopwords_title_es <- c(
    # Artículos
    "el","la","los","las","un","una","unos","unas",
    # Contracciones
    "al","del",
    # Conjunciones
    "y","e","ni","o","u",
    # Preposiciones
    "a","con","de","en","por","sin"
  )

  # Validaciones básicas
  if (!ent %in% colnames(df)) stop(glue::glue("La columna de entidad '{ent}' no existe en el data.frame."))
  if (!mun %in% colnames(df)) stop(glue::glue("La columna de municipio '{mun}' no existe en el data.frame."))
  if (!is.null(loc) && !loc %in% colnames(df)) stop(glue::glue("La columna de localidad '{loc}' no existe en el data.frame."))

  # Nombres de salida que se crearán
  nombre_nom_ent_mun <- "NOM_ENT_MUN"
  nombre_nom_ent_mun_loc <- "NOM_ENT_MUN_LOC"

  # # Evitar sobrescribir llaves ya existentes
  if (!sobreescribir_resultados) {

    if (is.null(loc)) {
      if (nombre_nom_ent_mun %in% colnames(df)) {
        cat(crayon::green(paste0(">> La columna '", nombre_nom_ent_mun, "' ya existe. No se realizó ninguna acción.\n")))
        return(df)
      }
    } else {
      if (nombre_nom_ent_mun_loc %in% colnames(df) || nombre_nom_ent_mun %in% colnames(df)) {
        cat(crayon::green(paste0(">> Las columnas de llaves ya existen (", paste(intersect(c(nombre_nom_ent_mun, nombre_nom_ent_mun_loc), colnames(df)), collapse = ", "), "). No se realizó ninguna acción.\n")))
        return(df)
      }
    }
  
  }


  # Preparar versión de entidad para nombres cortos si se solicita (no muta df)
  # CAMBIO: Renombrar variable a ent_vals_procesados para evitar colisión
  ent_vals_procesados <- df[[ent]]
  if (isTRUE(nombres_ent_cortos)) {
    ent_vals_procesados <- ent_vals_procesados |>
      str_replace_all(regex("COAHUILA DE ZARAGOZA", ignore_case = TRUE), "COAHUILA") |>
      str_replace_all(regex("VERACRUZ DE IGNACIO DE LA LLAVE", ignore_case = TRUE), "VERACRUZ") |>
      str_replace_all(regex("MICHOACÁN DE OCAMPO", ignore_case = TRUE), "MICHOACÁN")
  }

  # Función local para estandarizar texto igual que en procesar_claves_combinadas
  estandarizar_nombre <- function(x) {
    # Manejar NAs: si el valor es NA, devolver NA directamente
    if_else(is.na(x), NA_character_,
      x |> 
        str_squish() |> 
        str_to_title() |> 
        stringr::str_replace_all(
          regex(paste0("(?<=\\s)\\b(", paste(stopwords_title_es, collapse = "|"), ")\\b(?=\\s)"),
                ignore_case = TRUE)
        , function(y) tolower(y)) |> 
        stringr::str_replace_all(
          regex("\\b([IVXLCDM]+)\\b", ignore_case = TRUE),
          function(z) if (!is.na(z) && str_detect(z, regex("^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", ignore_case = TRUE))) toupper(z) else z
        )
    )
  }

  # Estandarizar entidad/municipio/localidad para luego convertir a snake_case
  ent_est <- estandarizar_nombre(ent_vals_procesados)
  mun_est <- estandarizar_nombre(df[[mun]])
  loc_est <- NULL
  if (!is.null(loc)) loc_est <- estandarizar_nombre(df[[loc]])

  # browser()

  # Generar las llaves de nombre (snake_case con transliteration Latin-ASCII)
  nom_ent_mun <- paste(
    snakecase::to_any_case(ent_est, case = "snake", transliterations = "Latin-ASCII"),
    snakecase::to_any_case(mun_est, case = "snake", transliterations = "Latin-ASCII"),
    sep = "_"
  )

  # Añadir a df
  out <- df %>% mutate(!!nombre_nom_ent_mun := nom_ent_mun)

  # Generar NOM_ENT_MUN_LOC y anexar si se pidió localidad
  if (!is.null(loc)) {
    nom_ent_mun_loc <- paste(
      snakecase::to_any_case(ent_est, case = "snake", transliterations = "Latin-ASCII"),
      snakecase::to_any_case(mun_est, case = "snake", transliterations = "Latin-ASCII"),
      snakecase::to_any_case(loc_est, case = "snake", transliterations = "Latin-ASCII"),
      sep = "_"
    )
    out <- out %>% mutate(!!nombre_nom_ent_mun_loc := nom_ent_mun_loc)
  }

  # Mensaje final informativo
  creadas <- setdiff(colnames(out), colnames(df))
  cat(crayon::green(paste0(">> Se agregaron las siguientes columnas: ", paste(creadas, collapse = ", "), "\n")))

  return(out)

}


#' Obtener datos geográficos de INEGI desde la base de datos local en PostgreSQL.
#'
#' @description
#' Función que extrae datos geográficos de INEGI desde una base de datos PostgreSQL.
#' Permite consultar información a nivel de entidades, municipios o localidades.
#'
#' @param level Character. Nivel de desagregación geográfica deseado. Acepta variaciones
#'   de "entidades", "municipios" o "localidades" (e.g., "estado", "muni", "localidad").
#'   Por defecto: "de las municipalidades apá".
#' @param envfile Character. Ruta al archivo de variables de entorno (.env).
#'   Por defecto: ".env".
#' @param nombres_ent_cortos Logical. Si TRUE, simplifica los nombres de entidades
#'   (e.g., "Coahuila de Zaragoza" → "Coahuila"). También actualiza las claves
#'   combinadas NOM_ENT_MUN y NOM_ENT_MUN_LOC con los nombres cortos.
#'   Por defecto: TRUE.
#' 
#' Otros parámetros de conexión a la base de datos PostgreSQL:
#' 
#' @param dbname Character. Nombre de la base de datos PostgreSQL. 
#'   Por defecto: variable de entorno DB_NAME_INEGI_RO.
#' @param host Character. Host del servidor PostgreSQL.
#'   Por defecto: variable de entorno DB_HOST_INEGI_RO.
#' @param port Character. Puerto del servidor PostgreSQL.
#'   Por defecto: variable de entorno DB_PORT_INEGI_RO.
#' @param user Character. Usuario de la base de datos PostgreSQL.
#'   Por defecto: variable de entorno DB_USER_INEGI_RO.
#' @param password Character. Contraseña del usuario de PostgreSQL.
#'   Por defecto: variable de entorno DB_PASSWORD_INEGI_RO.
#'
#' @return Un tibble con los datos geográficos del nivel especificado. La función
#'   imprime un mensaje en consola indicando el número de registros importados.
#'
#' @details
#' La función utiliza coincidencia difusa (Jaro-Winkler) mediante `select_inegi_level()`
#' para interpretar el parámetro `level` y seleccionar automáticamente el nivel geográfico
#' más apropiado. Requiere credenciales de PostgreSQL que pueden especificarse directamente
#' como parámetros o cargarse desde un archivo .env (DB_NAME_INEGI_RO, DB_HOST_INEGI_RO,
#' DB_PORT_INEGI_RO, DB_USER_INEGI_RO, DB_PASSWORD_INEGI_RO).
#'
#' La función se conecta al esquema 'geo_inegi' en la base de datos y cierra automáticamente
#' la conexión al finalizar usando `on.exit()`.
#'
#' @examples
#' \dontrun{
#' # Obtener datos de municipios
#' municipios <- get_geo_inegi(level = "municipios")
#'
#' # Obtener datos de entidades (acepta variaciones)
#' estados <- get_geo_inegi(level = "estados")
#'
#' # Obtener datos de localidades con archivo .env personalizado
#' localidades <- get_geo_inegi(level = "localidades", envfile = "config/.env")
#' 
#' # Obtener datos sin simplificar nombres de entidades
#' municipios_nombres_largos <- get_geo_inegi(level = "municipios", nombres_ent_cortos = FALSE)
#' }
#'
#' @importFrom DBI dbConnect dbDisconnect Id
#' @importFrom RPostgres Postgres
#' @importFrom dplyr tbl collect
#' @importFrom dotenv load_dot_env
#' @importFrom crayon green
#' @export
get_geo_inegi <- function(
  # Parámetros de la función
  level = "de las municipalidades apá",
  envfile = ".env",
  nombres_ent_cortos = TRUE,
  # Datos de la base de datos
  dbname = Sys.getenv("DB_NAME_INEGI_RO"),
  host = Sys.getenv("DB_HOST_INEGI_RO"),
  port = Sys.getenv("DB_PORT_INEGI_RO"),
  user = Sys.getenv("DB_USER_INEGI_RO"),
  password = Sys.getenv("DB_PASSWORD_INEGI_RO")
) {

    # Función interna para seleccionar nivel de INEGI con coincidencia difusa
    select_inegi_level <- function(text = "Los municipios apá") {

        # Paquetes
        pacman::p_load(stringdist, stringr)

        # Posibles niveles de INEGI
        posibles_niveles <- c("entidades", "municipios", "localidades")

        # Enrutador y estandarización de level: minúsculas y quitar última "s"
        level_raw <- tolower(text) |>
            str_replace_all("estado|edo|estatal|^esta?d?o?s?$", "entidades") |>
            str_replace_all("municipio|muni|municipal", "municipios") |>
            str_replace_all("localidad|localidades|población|poblacional", "localidades")
        # Usar Jaro-Winkler para encontrar el nivel más cercano (rankear y seleccionar el mejor)
        idx_nivel_seleccionado <- which.min(stringdist::stringdist(level_raw, posibles_niveles, method = "jw"))
        # Seleccionar el nivel correcto
        level = posibles_niveles[idx_nivel_seleccionado]

        # Retornar nivel seleccionado
        return(level)

    }

    # Cargar paquetes
    pacman::p_load(DBI, RPostgres, stringdist, tibble, ellmer)

    # # Cargar variables de entorno (Datos de conexión a la base de datos)
    # env_loaded <- tryCatch({
    #     dotenv::load_dot_env(file = envfile)
    #     TRUE
    # }, error = function(e) {
    #     warning("No se pudo cargar el archivo .env. Usando credenciales hardcodeadas.")
    #     FALSE
    # })
    
    # Enrutador de nivel de desagregación
    level <- level |> select_inegi_level()

    # Información de la base de datos:
    schema <- "geo_inegi"
    table <- level
    table_Id <- DBI::Id(schema = schema, table = table)

    # Conexión a la base de datos PostgreSQL
    con <- dbConnect(
        RPostgres::Postgres(),
        dbname = dbname,
        host = host,
        port = port,
        user = user,
        password = password
    )

    # Traer datos con dbplyr
    db_table <- tbl(con, table_Id) |> collect()

    # Procesamiento adicional: Nombres cortos de entidades
    if (nombres_ent_cortos) {
        cat("> Aplicando nombres cortos de entidades...\n")

        db_table <- db_table |>
        mutate(
        NOM_ENT = case_when(
        str_detect(NOM_ENT, "Coahuila de Zaragoza") ~ "Coahuila",
        str_detect(NOM_ENT, "Veracruz de Ignacio de la Llave") ~ "Veracruz",
        str_detect(NOM_ENT, "Michoacán de Ocampo") ~ "Michoacán",
        TRUE ~ NOM_ENT
        ),
        across(
        any_of(c("NOM_ENT_MUN", "NOM_ENT_MUN_LOC")),
        ~ str_replace_all(.x, "coahuila_de_zaragoza", "coahuila") |>
        str_replace_all("veracruz_de_ignacio_de_la_llave", "veracruz") |>
        str_replace_all("michoacan_de_ocampo", "michoacan")
        )
        )
    }

    # Desconectar al finalizar
    on.exit(dbDisconnect(con), add = TRUE)

    # Mensaje de cat, en color verde
    cat(paste0(">> Datos de '", crayon::green(level), "' cargados correctamente desde la base de datos.\n   Se han importado ", crayon::green(formatC(nrow(db_table), big.mark = ",")), " registros limpios con ", crayon::green("llaves de matching"), ".\n"))

    # Retorno de desarrollo
    return(db_table)

}




