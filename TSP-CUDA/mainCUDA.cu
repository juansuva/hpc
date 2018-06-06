#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;


struct ciudad
{
    float x;
    float y;
};



__global__ void
permutaciones_shared_register_big(int *matriz,int dimension,int *soluciones,int *d_num_aleatorios,int num_elementos_copiados,int inicio_copia)
{

    //Control de divergencia
    if((blockIdx.x * blockDim.x) + threadIdx.x < 10000)
    {
            //Numeros aletorios en memoria shared para mayor rendimiento
            __shared__ unsigned int num_aleatorios[4000];

            __syncthreads();

	    int distancia_total = 99999999;
	    int offset = (((blockIdx.x * blockDim.x) + threadIdx.x)*(dimension+1)); //Offset de la fila en la matriz soluciones

            //Copiamos el trozo correspondiente de la solucion
            unsigned int mi_solucion[85];

            for(int i = 0; i < num_elementos_copiados;i++)
            {
                mi_solucion[i] = soluciones[offset+inicio_copia+i];
            }

            __syncthreads();

            //Copiamos los numeros aleatorios en memoria shared
            for(int i = 0; i < 4000; i++)
            {
                    num_aleatorios[i] = d_num_aleatorios[i];
            }

            __syncthreads();

	    //Realizamos 90000 permutaciones para intentar mejorar la solucion greedy
	    for(int i = 0; i < 90000; i++)
	    {

                 __syncthreads();

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

        __syncthreads();
        //Guardamos el coste en la ultima posicion de cada columna

        for(int i = 0; i < num_elementos_copiados;i++)
        {
            soluciones[offset+i+inicio_copia] = mi_solucion[i];
        }

       //Unimos las ciudades de cada trozo es decir
        //ultima del primer trozo - primera del segundo trozo etc
        if(inicio_copia > 0 && num_elementos_copiados >= 84)
        {
            distancia_total += matriz[soluciones[offset+inicio_copia+85]*dimension + mi_solucion[0]];
        }

        //Unimos la ultima con la primera
        if(num_elementos_copiados <= 84  && inicio_copia != 0)
        {
            distancia_total += matriz[mi_solucion[num_elementos_copiados-1]*dimension + soluciones[offset]];
        }

        soluciones[offset+dimension] += distancia_total;
    }

}


__global__ void
permutaciones(int *matriz,int dimension,int *soluciones,int *num_aleatorios)
{

    //Control de divergencia
    if((blockIdx.x * blockDim.x) + threadIdx.x < 10000)
    {

	    int distancia_total = 99999999;
	    int offset = (((blockIdx.x * blockDim.x) + threadIdx.x)*(dimension+1)); //Offset de la fila en la matriz soluciones


	    //Realizamos 100000 permutaciones para intentar mejorar la solucion inicial
	    for(int i = 0; i < 100000; i++)
	    {

                 __syncthreads();

		int posicion_1 = num_aleatorios[i*2] + (offset);
		int posicion_2 = num_aleatorios[(i*2)+1] + (offset);
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
		for(int j = 0; j < dimension-2; j++)
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

        __syncthreads();

        //Unimos la primera y la ultima ciudad
        distancia_total += matriz[soluciones[offset]*dimension + soluciones[offset+dimension-1]];

        //Guardamos el coste en la ultima posicion de cada columna
        soluciones[offset+dimension] = distancia_total;
    }

}


//Podemos almacenar maximo 85x85 ciudades en la memoria __shared__ puesto que
//tenemos una memoria __shared__ de 16Kb => 16384 elementos de 8 bits
// 85x85x8 = 16200
__global__ void
permutaciones_shared_register(int *d_matriz,int dimension,int *soluciones,int *num_aleatorios)
{

    //Control de divergencia
    if((blockIdx.x * blockDim.x) + threadIdx.x < 10000)
    {
            //Matriz de distancias en memoria compartida para mayor rendimiento
            extern __shared__ unsigned char matriz[];

            if(!threadIdx.x) //Solo thread 0 inicializa la matriz compartida
            {
                for(int i = 0; i < dimension*dimension; i++)
                {
                    matriz[i] = d_matriz[i];
                }
            }

            __syncthreads();

	    int distancia_total = 99999999;
	    int offset = (((blockIdx.x * blockDim.x) + threadIdx.x)*(dimension+1)); //Offset de la fila en la matriz soluciones

            unsigned char mi_solucion[85];

            for(int i = 0; i < dimension;i++)
            {
                mi_solucion[i] = soluciones[offset+i];
            }

            __syncthreads();

	    //Realizamos 100000 permutaciones para intentar mejorar la solucion greedy
	    for(int i = 0; i < 100000; i++)
	    {

                 __syncthreads();

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

        __syncthreads();
        //Guardamos el coste en la ultima posicion de cada columna

        for(int i = 0; i < dimension;i++)
        {
            soluciones[offset+i] = mi_solucion[i];
        }

        //Unimos la primera y la ultima ciudad
        distancia_total += matriz[mi_solucion[0]*dimension + mi_solucion[dimension-1]];

        soluciones[offset+dimension] = distancia_total;
    }

}




__global__ void
permutaciones_register(int *matriz,int dimension,int *soluciones,int *num_aleatorios)
{

    //Control de divergencia
    if((blockIdx.x * blockDim.x) + threadIdx.x < 10000)
    {

        int distancia_total = 99999999;
        int offset = (((blockIdx.x * blockDim.x) + threadIdx.x)*(dimension+1)); //Offset de la fila en la matriz soluciones

        unsigned char mi_solucion[90]; //Maximo 90 puesto que en compute capability 2.x y 3.x tenemos máximo 63 registros
                                        //y debemos tener en cuenta el resto de variables registradas

        for(int i = 0; i < dimension;i++)
        {
            mi_solucion[i] = soluciones[offset+i];
        }
        
        __syncthreads();

        //Realizamos 100000 permutaciones para intentar mejorar la solucion greedy
        for(int i = 0; i < 100000; i++)
        {

                __syncthreads();

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

        __syncthreads();
        //Guardamos el coste en la ultima posicion de cada columna

        for(int i = 0; i < dimension;i++)
        {
            soluciones[offset+i] = mi_solucion[i];
        }


        //Unimos la primera y la ultima ciudad
        distancia_total += matriz[mi_solucion[0]*dimension + mi_solucion[dimension-1]];

        soluciones[offset+dimension] = distancia_total;
    }

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

    cudaError_t err;

    //Semilla
    srand(111);

    const int NUMERO_SOLUCIONES = 10000;

    string str;
    int dimension;
    int *h_matriz; //Matriz de distancias
    int *h_soluciones; //Matriz de soluciones
    int *h_num_aleatorios; //Vector de numeros aleatorios

    vector<int> soluciones_std;

    f >> str; //Nos saltamos la palabra de cabecera
    f >> dimension;

    ciudad *ciudades;
    ciudades = new ciudad[dimension];


    //Inicializamos las matrices
    h_matriz = (int *)malloc((dimension*dimension)*sizeof(int));
    h_soluciones = (int *)malloc((NUMERO_SOLUCIONES*(dimension+1))*sizeof(int)); //+1 Columna extra para el coste de cada solucion

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
            h_matriz[i*dimension + j] = distancia;
        }
    }

    //Generamos vector de numeros aleatorios 100000 elementos (90000*2)
    h_num_aleatorios = (int *)malloc((200000)*sizeof(int));

    for(int i = 0; i < 200000; i++)
    {
        h_num_aleatorios[i] = rand() % dimension;
    }

    //Generamos 10000 soluciones iniciales aleatorias
    for(int i = 0; i < dimension; i++)
    {
        soluciones_std.push_back(i);
    }

    for(int i = 0; i < (dimension+1)*NUMERO_SOLUCIONES; i += dimension + 1)
    {
            std::random_shuffle(soluciones_std.begin(),soluciones_std.end());

            for(int j = 0; j < soluciones_std.size(); j++)
            {
                h_soluciones[i + j] = soluciones_std.at(j);
            }
    }


    //Copiamos los vectores al device (host to device)
    int *d_matriz = NULL; //Matriz de distancias
    int *d_soluciones = NULL; //Matriz de soluciones
    int *d_num_aleatorios; //Vector numeros aleatorios

    err = cudaMalloc((void **)&d_matriz, (dimension*dimension)*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector matriz (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_soluciones, (NUMERO_SOLUCIONES*(dimension+1))*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector soluciones (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_num_aleatorios, 200000*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector numeros aleatorios (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(d_matriz, h_matriz, (dimension*dimension)*sizeof(int), cudaMemcpyHostToDevice);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector matriz from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(d_num_aleatorios, h_num_aleatorios, 200000*sizeof(int), cudaMemcpyHostToDevice);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector numeros aleatorios from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(d_soluciones, h_soluciones, (NUMERO_SOLUCIONES*(dimension+1))*sizeof(int), cudaMemcpyHostToDevice);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector soluciones from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }




    double start_time_calculo = omp_get_wtime();

    //Lanzamos 40 bloques con 290 hebras por bloque = 10000 hebras

    //Mayor que 90 ciudades no cabe en registros ni en memoria shared (método mas lento)
    if(dimension > 90){
       // permutaciones<<<40, 290>>>(d_matriz, dimension, d_soluciones,d_num_aleatorios);
        int inicio_copia = 0;
        int num_elementos_copiados = 0;

        for(int i = 0; i < (dimension/85) +  1; i++ )
        {
            num_elementos_copiados = 0;

            for(int j = inicio_copia; j < (dimension*i)+85; j++)
            {
                if(j >= dimension || num_elementos_copiados == 84)
                    break;

                num_elementos_copiados++;
            }
            permutaciones_shared_register_big<<<40, 290>>>(d_matriz, dimension, d_soluciones,d_num_aleatorios,num_elementos_copiados,inicio_copia);
            cudaDeviceSynchronize(); //Barrera

            inicio_copia = (i+1)*85;
        }
    }
    else if(dimension <= 85) //Cabe en registros y memoria shared (el mas rápido)
        permutaciones_shared_register<<<40, 290,dimension*dimension*sizeof(unsigned char)>>>(d_matriz, dimension, d_soluciones,d_num_aleatorios);
    else if(dimension > 85 && dimension <= 90) //Solo cabe en registros
        permutaciones_register<<<40, 290>>>(d_matriz, dimension, d_soluciones,d_num_aleatorios);

    cudaDeviceSynchronize(); //Barrera

    double  end_time_calculo = omp_get_wtime() - start_time_calculo;

    cout << "Tiempo calculo: " << end_time_calculo << " segundos" << endl; //Imprimir el tiempo empleado

    double start_time_copia = omp_get_wtime();

    //Copiamos las soluciones del device al host
    err = cudaMemcpy(h_soluciones, d_soluciones, (NUMERO_SOLUCIONES*(dimension+1))*sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector soluciones from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double end_time_copia = omp_get_wtime() - start_time_copia;

    cout << "Tiempo de copia soluciones (device to host): " << end_time_copia << " segundos" << endl; //Imprimir el tiempo empleado en copiar las soluciones

    //Nos quedamos con el mejor coste y por tanto la mejor solucion
    int menor_coste = h_soluciones[dimension];
    int indice_solucion = 0;
    for(int i = dimension; i < (NUMERO_SOLUCIONES*dimension); i+= dimension+1)
    {
        if(h_soluciones[i] < menor_coste)
        {
            indice_solucion = i - dimension; //Para situarnos al principio de la fila
            menor_coste = h_soluciones[i]; //Actualizamos el menor coste
        }
    }


    //Imprimimos la mejor solucion y su coste
    cout << "Distancia total de la mejor solucion: " << h_soluciones[indice_solucion+dimension] << endl;
    for(int i = indice_solucion; i < indice_solucion+dimension; i++)
    {
        cout << h_soluciones[i] + 1 << endl;
    }

    //Liberamos memoria del device
    err = cudaFree(d_soluciones);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector soluciones (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_matriz);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector matriz (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_num_aleatorios);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector numeros aleatorios (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_matriz);
    free(h_soluciones);
    free(h_num_aleatorios);

    return 0;

}
