# Multiplicaciones de matrices


## Tools  


## Crear matriz

[makematriz.c](https://github.com/juansuva/hpc/blob/master/matriz/tools/makematriz.c) es un generador de matrices.


Donde Se deben enviar por parámetro el nombre del archivo de salida, los dos primeros númerosson las filas y columnas respectivas de la matriz,

### Para  compilarlo

```bash
gcc makematriz.c -o create.out
```

### Para ejecutarlo


```bash
./create.out matriz1.in matriz2.in
```



## Comparar archivos
[compare.c](https://github.com/juansuva/hpc/blob/master/matriz/tools/compare.c) es un comparador de archivos.

### Para  compilarlo

```bash
gcc compare.c -o compare.out
```

### Para ejecutarlo


```bash
./compare.out matrix1.in matrix2.in
```
