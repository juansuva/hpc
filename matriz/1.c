#include <stdio.h>
#include <stdlib.h>

typedef char* string;


void guardar(float *resultado, int size, string file_name) {
  FILE *f = fopen(file_name, "w");
  fprintf(f, "%d\n", size);
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  int i, j;
  for (i = 0; i < size; i++) {
    printf("%f\n",resultado[i] );
    if (i + 1 == size) {
      fprintf(f, "%.2f", resultado[i]);
      printf("%s\n","st" );
    } else {
      fprintf(f, "%.2f,", resultado[i]);
    }
  }
  fclose(f);
}



void make(float *M,int size){
  for(int t=0;t<size;t++){
    float p = rand()%100;
    M[t]=p;
    printf("%2.f\n",M[t]);
  }
}

void imprime(float *M,int size){
  for(int t=0;t<size;t++){
    printf("%s","la suma es");
    printf("%2.f\n",M[t]);
  }
}


void suma(float *M, float *M2, float *resultado,int size){
  for(int i=0;i<size;i++){
    resultado[i]=M[i]+M2[i];

  }
}

void main(){
  int num,num2=0;
  printf ("¿Cuántos elementos quiere en el vector 1? ");
  scanf ("%d", &num);
  float *M = (float*)malloc(num+1*sizeof(float));
  if (M==NULL)
      {
      perror("Problemas reservando memoria");
      exit (1);
      }
  printf ("¿Cuántos elementos quiere en el vector 2? ");
  scanf ("%d", &num2);
  while(num>num2)
      {
      perror("el vector dos debe ser mas grande que el vector 1");
      scanf ("%d", &num2);
      }
  float *M2 = (float*)malloc(num2+1*sizeof(float));
  if (M2==NULL)
      {
      perror("Problemas reservando memoria");
      exit (1);
      }
  float *resultado = (float*)malloc(num2*sizeof(float));
  make(M,num);
  make(M2,num2);
  suma(M,M2,resultado,num);
  imprime(resultado,num);
  guardar(resultado,num,"result.csv");
  guardar(M,num,"vector1.csv");
  guardar(M2,num2,"vector2.csv");
}
