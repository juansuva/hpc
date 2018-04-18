#include <stdio.h>
#include <stdlib.h>
#include <string>

void make(int row, int col) {
  float a = 5.0;

  FILE *stream;
  stream = fopen("matriz1.txt", "r");

  for(int i = 0; i < row; i++) {
    for(int j = 0; j < col; j++) {
      a=(float)rand() / (float)(RAND_MAX / a);
      fprintf(stream, "%f",a);
    }
    fprintf(stream, "%s\n","");
  }
  fclose(stream);

}

int main(){
 make(50,100);
 return 1;
}
