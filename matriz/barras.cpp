#include <stdlib.h>
#include <stdio.h>
#include <conio2> //Linkear con Lconio.a

int i, j;

float prom1=8.50, prom2=6.33, prom3=9.10;

main(){
//system("Mode con: Lines=53");
for(i =0; i<42; i++)
{
gotoxy(5, i+1);
printf("|");
if ((40-i)%4==0) {
gotoxy(2, i+1);
printf("%2d-", (40 - i) / 4);
}
}

for(i = 5; i<75; i++)
{
gotoxy(i, 41);
printf("-");
if (i == 10) { gotoxy(i, 42); printf("Promedio1"); }
if (i == 20) { gotoxy(i, 42); printf("Promedio2"); }
if (i == 30) { gotoxy(i, 42); printf("Promedio3"); }
}

textbackground(2);
for(i = (41-(prom1*4)); i<41; i++){gotoxy(10,i); cprintf(" ");}
textbackground(1);
for(i = (41-(prom2*4)); i<41; i++){gotoxy(20,i); cprintf(" ");}
textbackground(3);
for(i = (41-(prom3*4)); i<41; i++){gotoxy(30,i); cprintf(" ");}


textbackground(0);
gotoxy(1,48);
system("Pause");
return 0;
}
