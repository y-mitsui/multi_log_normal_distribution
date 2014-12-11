CFLAGS=-Wall -g `pkg-config --cflags standard` `pkg-config --cflags glib-2.0` `pkg-config --cflags apr-1` `xml2-config --cflags`
LOADLIBES=`gsl-config --libs` `pkg-config --libs standard` `pkg-config --libs glib-2.0` `pkg-config --libs apr-1``xml2-config --libs`
example2:
