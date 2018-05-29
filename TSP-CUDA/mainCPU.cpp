#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>


using namespace std;


struct ciudad
{
    float x;
    float y;
};

void permutaciones_big(int *matriz,int dimension,int ciudad_inicial,int *soluciones,int *d_num_aleatorios,int num_elementos_copiados,int inicio_copia)
{

    int distancia_total = 99999999;
    int offset = ((ciudad_inicial*(dimension+1))); //Offset de la fila en la matriz soluciones

    unsigned int mi_solucion[45];


    
    for(int i = 0; i < num_elementos_copiados;i++)
    {
        mi_solucion[i] = soluciones[offset+inicio_copia+i];
    }

    unsigned int num_aleatorios[4000];

     //Copiamos los numeros aleatorios en memoria shared
    for(int i = 0; i < 4000; i++)
    {
        num_aleatorios[i] = d_num_aleatorios[i];
    }

    //Realizamos 50000 permutaciones para intentar mejorar la solucion
    for(int i = 0; i < 50000; i++)
    {
        int posicion_1 = num_aleatorios[(i*2) % num_elementos_copiados] % num_elementos_copiados;
        int posicion_2 = num_aleatorios[(i*2 + 1) % num_elementos_copiados] % num_elementos_copiados;
        int temp = 0;

        //Guardamos los elementos temporalmente por si la solucion no es mejor
        //poder restaurar los elementos
        int elemento_1 = mi_solucion[posicion_1];
        int elemento_2 = mi_solucion[posicion_2];

        //Realizamos el swap de ambos elementos
        temp = mi_solucion[posicion_1];
        mi_solucion[posicion_1] = mi_solucion[posicion_2];
        mi_solucion[posicion_2] = temp;

        //Calculamos el coste la nueva solucion
        int nueva_distancia_total = 0;
        for(int j = 0; j < num_elementos_copiados-1; j++)
        {
            nueva_distancia_total += matriz[mi_solucion[j]*dimension + mi_solucion[j+1]];
        }

        //Comprobamos si la solucion mejora
        if(nueva_distancia_total < distancia_total) //La solucion obtenida mejora
        {
            distancia_total = nueva_distancia_total;
        }
        else //No mejora, restauramos los elementos
        {
            mi_solucion[posicion_1] = elemento_1;
            mi_solucion[posicion_2] = elemento_2;
        }
    }

    //Guardamos el coste en la ultima posicion de cada columna
    for(int i = 0; i < num_elementos_copiados;i++)
    {
        soluciones[offset+i+inicio_copia] = mi_solucion[i];
    }

    //Unimos las ciudades de cada trozo es decir
    //ultima del primer trozo - primera del segundo trozo etc
    if(inicio_copia > 0 && num_elementos_copiados >= 44)
    {
        distancia_total += matriz[soluciones[offset+inicio_copia+45]*dimension + mi_solucion[0]];
    }
    
    //Unimos la ultima con la primera
    if(num_elementos_copiados <= 44  && inicio_copia != 0)
    {
        distancia_total += matriz[mi_solucion[num_elementos_copiados-1]*dimension + soluciones[offset]];
    }
    
    soluciones[offset+dimension] += distancia_total;

}

void permutaciones_50(int *matriz,int dimension,ciudad *ciudades,int ciudad_inicial,int *soluciones,int *num_aleatorios)
{

    int distancia_total = 99999999;
    int offset = ((ciudad_inicial*(dimension+1))); //Offset de la fila en la matriz soluciones
    unsigned char mi_solucion[dimension];

    for(int i = 0; i < dimension;i++)
    {
        mi_solucion[i] = soluciones[offset+i];
    }

    //Realizamos 100000 permutaciones para intentar mejorar la solucion
    for(int i = 0; i < 100000; i++)
    {
        int posicion_1 = num_aleatorios[i*2];
        int posicion_2 = num_aleatorios[(i*2)+1];
        int temp = 0;

        //Guardamos los elementos temporalmente por si la solucion no es mejor
        //poder restaurar los elementos
        int elemento_1 = mi_solucion[posicion_1];
        int elemento_2 = mi_solucion[posicion_2];

        //Realizamos el swap de ambos elementos
        temp = mi_solucion[posicion_1];
        mi_solucion[posicion_1] = mi_solucion[posicion_2];
        mi_solucion[posicion_2] = temp;

        //Calculamos el coste la nueva solucion
        int nueva_distancia_total = 0;
        for(int j = 0; j < dimension-1; j++)
        {
            nueva_distancia_total += matriz[mi_solucion[j]*dimension + mi_solucion[j+1]];
        }

        //Comprobamos si la solucion mejora
        if(nueva_distancia_total < distancia_total) //La solucion obtenida mejora
        {
            distancia_total = nueva_distancia_total;
        }
        else //No mejora, restauramos los elementos
        {
            mi_solucion[posicion_1] = elemento_1;
            mi_solucion[posicion_2] = elemento_2;
        }
    }

    //Guardamos el coste en la ultima posicion de cada columna
    for(int i = 0; i < dimension;i++)
    {
        soluciones[i+offset] = mi_solucion[i];
    }

     //Unimos la primera y la ultima ciudad
    distancia_total += matriz[mi_solucion[0]*dimension + mi_solucion[dimension-1]];

    soluciones[offset+dimension] = distancia_total;

}



void permutaciones(int *matriz,int dimension,ciudad *ciudades,int ciudad_inicial,int *soluciones,int *num_aleatorios)
{

    int distancia_total = 99999999;
    int offset = ((ciudad_inicial*(dimension+1))); //Offset de la fila en la matriz soluciones

    //Realizamos 100000 permutaciones para intentar mejorar la solucion
    for(int i = 0; i < 100000; i++)
    {
        int posicion_1 = num_aleatorios[i*2] + offset;
        int posicion_2 = num_aleatorios[(i*2)+1] + offset;
        int temp = 0;

        //Guardamos los elementos temporalmente por si la solucion no es mejor
        //poder restaurar los elementos
        int elemento_1 = soluciones[posicion_1];
        int elemento_2 = soluciones[posicion_2];

        //Realizamos el swap de ambos elementos
        temp = soluciones[posicion_1];
        soluciones[posicion_1] = soluciones[posicion_2];
        soluciones[posicion_2] = temp;

        //Calculamos el coste la nueva solucion
        int nueva_distancia_total = 0;
        for(int j = 0; j < dimension-1; j++)
        {
            nueva_distancia_total += matriz[soluciones[offset+j]*dimension + soluciones[offset+j+1]];
        }

        //Comprobamos si la solucion mejora
        if(nueva_distancia_total < distancia_total) //La solucion obtenida mejora
        {
            distancia_total = nueva_distancia_total;
        }
        else //No mejora, restauramos los elementos
        {
            soluciones[posicion_1] = elemento_1;
            soluciones[posicion_2] = elemento_2;
        }
    }

    //Unimos la primera y la ultima ciudad
    distancia_total += matriz[soluciones[offset]*dimension + soluciones[offset+dimension-1]];

    //Guardamos el coste en la ultima posicion de cada columna
    soluciones[offset+dimension] = distancia_total;

}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout << "Uso: ./tsp archivo.tsp" << endl;
        return 0;
    }

    ifstream f(argv[1]);

    if(!f)
    {
        cout << "Error al abrir el fichero " << argv[1] << " .El fichero no existe" << endl;
        return 0;
    }

    //Semilla
    srand(111);

    const int NUMERO_SOLUCIONES = 10000;

    string str;
    int dimension;
    int *matriz; //Matriz de distancias
    int *soluciones; //Matriz de soluciones
    int *num_aleatorios; //Vector de numeros aleatorios


    vector<int> soluciones_std;

    f >> str; //Nos saltamos la palabra de cabecera
    f >> dimension;

    ciudad *ciudades;
    ciudades = new ciudad[dimension];


    //Inicializamos las matrices
    matriz = new int[dimension*dimension];
    soluciones = new int[NUMERO_SOLUCIONES*(dimension+1)]; //+1 Columna extra para el coste de cada solucion

    //Leemos el indice de ciudad
    int i = 0;
    while(!f.eof())
    {

        f >> str;
        f >> ciudades[i].x;
        f >> ciudades[i].y;
        i++;
    }

    //Calculamos la matriz de distancias
    for(int i = 0; i < dimension; i++)
    {
        for(int j = 0; j < dimension; j++)
        {
            int distancia = round(sqrt(((ciudades[i].x - ciudades[j].x)*(ciudades[i].x - ciudades[j].x)) + ((ciudades[i].y - ciudades[j].y)*(ciudades[i].y - ciudades[j].y))));
            matriz[i*dimension + j] = distancia;
        }
    }

    //Generamos vector de numeros aleatorios 100000 elementos (50000*2)
    num_aleatorios = new int[200000];

    for(int i = 0; i < 200000; i++)
    {
        num_aleatorios[i] = rand() % dimension;
    }


    //Generamos 1000 soluciones iniciales aleatorias
    for(int i = 0; i < dimension; i++)
    {
        soluciones_std.push_back(i);
    }

    for(int i = 0; i < (dimension+1)*NUMERO_SOLUCIONES; i += dimension + 1)
    {
            std::random_shuffle(soluciones_std.begin(),soluciones_std.end()); //Shuffle de ciudades

            for(int j = 0; j < soluciones_std.size(); j++)
            {
                soluciones[i + j] = soluciones_std.at(j);
            }
    }


    double start_time_calculo = omp_get_wtime();

    //Lanzamos la heuristica 10000 veces cambiando la ciudad de partida
    if(dimension <=50)
    {
        for(int i = 0; i < NUMERO_SOLUCIONES; i++)
            permutaciones_50(matriz,dimension,ciudades,i,soluciones,num_aleatorios);
    }
    else if( dimension > 50)
    {
        for(int x = 0; x < NUMERO_SOLUCIONES; x++)
        {
            int inicio_copia = 0;
            int num_elementos_copiados = 0;

            for(int i = 0; i < (dimension/45) +  1; i++ )
            {
                num_elementos_copiados = 0;

                for(int j = inicio_copia; j < (dimension*i)+45; j++)
                {
                    if(j >= dimension || num_elementos_copiados == 44)
                    {
                //        cout << "BREAK " << endl;
                        break;
                    }

                    num_elementos_copiados++;
                //    cout << j << endl;
                }

              //  cout << "HEBRA : " << j << endl;

               permutaciones_big(matriz,dimension,x,soluciones,num_aleatorios,num_elementos_copiados,inicio_copia);

               inicio_copia = (i+1)*45;
            }
        }
    }

    double end_time_calculo = omp_get_wtime() - start_time_calculo;

    cout << "Tiempo calculo: " << end_time_calculo << " segundos" << endl; //Imprimir el tiempo empleado


    //Nos quedamos con el mejor coste y por tanto la mejor solucion
    int menor_coste = soluciones[dimension];
    int indice_solucion = 0;
    for(int i = dimension; i < (NUMERO_SOLUCIONES*dimension); i+= dimension+1)
    {
        if(soluciones[i] < menor_coste)
        {
            indice_solucion = i - dimension; //Para situarnos al principio de la fila
            menor_coste = soluciones[i]; //Actualizamos el menor coste
        }
    }

    //Imprimimos la mejor solucion y su coste
    cout << "Distancia total de la mejor solucion: " << soluciones[indice_solucion+dimension] << endl;
    for(int i = indice_solucion; i < indice_solucion+dimension; i++)
    {
        cout << soluciones[i] + 1 << endl;
    }

    delete matriz;
    delete soluciones;
    delete num_aleatorios;

}
