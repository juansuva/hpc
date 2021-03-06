#include <stdio.h>
#include <stdlib.h>

typedef char* string;

void make(float **M, int row, int col) {
  float a = 5.0;


  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      M[i][j] = (float)rand() / (float)(RAND_MAX / a);

    }

  }

}

void imprime(float **M, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%.2f ", M[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}


void multiplica(float **M1, float **M2, int fil1, int col1,int fil2, int col2,float **resultado){


  for (int i = 0; i <fil1; i++) {

    for (int j = 0; j < col2; j++) {

      float acu=0;
      for (int con= 0; con < col1; con++) {


        acu=acu+M1[i][con]*M2[con][j];

        resultado[i][j]=acu;
      }
    }
  }

}

void guardar(float **resultado, int col, int fil, string file_name) {
  FILE *f = fopen(file_name, "w");
  fprintf(f, "%d,%d\n", fil,col);
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  int i, j;
  for (i = 0; i < fil; i++) {
    for(j=0;j<col;j++){
        printf("%f\n",resultado[i][j] );
        if (i + 1 == col) {
          fprintf(f, "%.2f", resultado[i][j]);
          printf("%s\n","st" );
        } else {
          fprintf(f, "%.2f,", resultado[i][j]);
        }

      }
      fprintf(f,"%s\n","" );
  }
  fclose(f);
}


int main() {

  int row, col,col1,col2,fil1,fil2;
  printf ("¿cuantos elementos quiere en la fila del vector 1? ");
  scanf ("%d", &fil1);
  printf ("¿cuantos elementos quiere en la columna del vector 1? ");
  scanf ("%d", &col1);
  printf ("¿cuantos elementos quiere en la fila del vector 2? ");
  scanf ("%d", &fil2);
  printf ("¿cuantos elementos quiere en la columna del vector 2? ");
  scanf ("%d", &col2);
  printf("%d,%d\n",fil1,col2 );

  while(col1!=fil2){

    printf ("Usted solo puede multiplicar dos matrices si sus dimensiones son compatibles , lo que significa que el número de columnas en la primera matriz es igual al número de filas en la segunda matriz ");
    printf("%d,%d\n",fil1,col2 );
    printf("%s \n", "el numero de columnas  de la primera matriz es" );
    printf("%d\n",col1 );
    scanf ("%d", &fil2);
  }

  //int row, col;
//  string file_name = argv[1];
//  printf("File name: %s\n", file_name);
  //scanf("%d %d", &row, &col);
  float **M1;
   M1 = (float **)malloc (fil1*sizeof(float *));
  for (int i=0;i<fil1;i++){
      M1[i] = (float *) malloc (col1*sizeof(float));
  }

  float **M2;
   M2 = (float **)malloc (fil2*sizeof(float *));
  for (int i=0;i<fil2;i++){
      M2[i] = (float *) malloc (col2*sizeof(float));
  }

  float **resultado;
   resultado = (float **)malloc (fil1*sizeof(float *));
  for (int i=0;i<fil1;i++){
      resultado[i] = (float *) malloc (col2*sizeof(float));
  }


  make(M1, fil1, col1);
  make(M2, fil2, col2);
  imprime(M1, fil1, col1);
  imprime(M2, fil2, col2);
  multiplica(M1,M2,fil1,col1,fil2,col2,resultado);
  imprime(resultado, fil1,col2);
  guardar(resultado,col2,fil1,"result.csv");
  //writeMatrix(M1, row, col, "matriz");
  return 0;
}
