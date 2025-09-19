#!/bin/sh
set -eu

# Ejecutar como root
if [ "$(id -u)" -ne 0 ]; then
  echo "[ERROR] Ejecuta como root (sudo)." 1>&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo ">>> Actualizando indices..."
apt-get update -y
apt-get upgrade -y || true

echo ">>> Instalando dependencias, una por linea..."

# Monitor de uso del sistema (CPU, memoria, disco, red)
apt-get install -y atop

# Monitor de ancho de banda en interfaces de red
apt-get install -y bmon

# Entorno de terminal basado en tmux, con statusbar
apt-get install -y byobu

# Coloreador de logs para facilitar lectura
apt-get install -y ccze

# Efecto Matrix (lluvia de caracteres en pantalla)
apt-get install -y cmatrix

# GNU Awk: lenguaje de procesamiento de textos/archivos
apt-get install -y gawk

# Conversor de imagenes JPG a ASCII en consola
apt-get install -y jp2a

# Archivos comunes de configuracion para libconfuse
apt-get install -y libconfuse-common

# Libreria de analisis de archivos de configuracion
apt-get install -y libconfuse2

# Libreria de eventos asincronos (dependencia central)
apt-get install -y libevent-core-2.1-7t64 || apt-get install -y libevent-core-2.1-7 || true

# Modulo Perl para trabajar con pseudo-terminales
apt-get install -y libio-pty-perl

# Modulo Perl para ejecutar procesos y comandos externos
apt-get install -y libipc-run-perl

# Libreria para manejar senales en C
apt-get install -y libsigsegv2

# Libreria Perl para formatear duraciones de tiempo
apt-get install -y libtime-duration-perl

# Libreria de syscalls modernos para E/S asincronica (io_uring)
apt-get install -y liburing2

# Coleccion de utilidades adicionales para shell (ej. sponge, parallel, ts)
apt-get install -y moreutils

# Herramienta para enviar texto a servicios de pastebin desde la terminal
apt-get install -y pastebinit

# Utilidad de busqueda rapida basada en mlocate
apt-get install -y plocate

# Soporte de menus en modo texto (newt library para Python3)
apt-get install -y python3-newt

# Libreria Python para medir uso de CPU, memoria, disco, red
apt-get install -y python3-psutil

# Libreria Python para crear interfaces de texto interactivas
apt-get install -y python3-urwid

# Libreria Python que maneja ancho de caracteres (ej. Unicode)
apt-get install -y python3-wcwidth

# Herramienta para ejecutar un comando a la vez, evita duplicados
apt-get install -y run-one

# Monitor de velocidad de red en consola
apt-get install -y speedometer

# Multiplexor de terminal (division de ventanas/persistencia)
apt-get install -y tmux

# Listado de directorios en forma de arbol
apt-get install -y tree

echo ">>> Listo. Dependencias instaladas ✅"
