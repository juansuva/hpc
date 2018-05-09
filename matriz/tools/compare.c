#include <stdio.h>
#include <stdlib.h>

typedef char* string;

int compareFiles(string archivo1, string archivo2) {
  FILE *f1 = fopen(archivo1, "r");
  FILE *f2 = fopen(archivo2, "r");
  int ans = 1;
  char c1, c2;
  if (f1 && f2) {
    while ((c1 = fgetc(f1)) != EOF && (c2 = fgetc(f2)) != EOF) {
      if (c1 != c2) {
        ans = 0;
        break;
      }
    }
  } else {
    printf("Error al abrir archivo!\n");
    ans = 0;
  }
  fclose(f1);
  fclose(f2);
  return ans;
}

int main(int argc, char** argv) {
  if (argc =! 3) {
    printf("Must be called with the name of the two files\n");
    return 1;
  }
  string archivo1 = argv[1];
  string archivo2 = argv[2];
  printf("File names: %s, %s\n", archivo1, archivo2);
  if (compareFiles(archivo1, archivo2)) printf("los archivos son iguales \n");
  else printf("los archivos no son iguales\n");
  return 0;
}
