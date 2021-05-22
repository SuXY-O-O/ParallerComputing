#include <pthread.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <time.h>

#define MAT_0 500
#define MAT_1 500
#define MAT_2 500
#define APATH "./mats/A"
#define BPATH "./mats/B"
#define CPATH "./mats/C"

void print_mat(const char *file_name, int n1, int n2, double *mat)
{
    printf("------ %s: %d*%d Matrix ------\n", file_name, n1, n2);
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            printf("%.4f  ", *(mat + i * n2 + j)); // mat[i,j]
        }
        printf("\n");
    }

    FILE *file;
    if (!(file = fopen(file_name, "w")))
    {
        printf("Can't open file %s\n", file_name);
        return;
    }
    fwrite(&n1, sizeof(int), 1, file);
    fwrite(&n2, sizeof(int), 1, file);
    fwrite(mat, sizeof(double), n1 * n2, file);
    fclose(file);

    return;
}

void make_mat(const char *file_name, int m, int n, double *mat)
{
    int bufsize = sizeof(int) * 2 + sizeof(double) * m * n;
    double *a;
    a = (double *)malloc(bufsize);

    ((int *)a)[0] = m;
    ((int *)a)[1] = n;

    double *ptr = (double *)((int *)a + 2);

    srand48(time(NULL)); // Use time as a seed
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            *(ptr + i * n + j) = drand48();
            *(mat + i * n + j) = *(ptr + i * n + j);
        }
    }

    FILE *file;
    if (!(file = fopen(file_name, "w")))
    {
        printf("Can't open file %s\n", file_name);
        return;
    }
    fwrite(a, sizeof(char), bufsize, file);
    fclose(file);

    return;
}

struct threadArg
{
    int tid;
    double *B;
    double *A_row;
    double *C_row;
    int numthreads;
};

void *worker(void *arg)
{
    int i, j;
    struct threadArg *myarg = (struct threadArg *)arg;
    for (i = myarg->tid; i < MAT_2; i += myarg->numthreads)
    {
        myarg->C_row[i] = 0.0;
        for (j = 0; j < MAT_1; j++)
        {
            myarg->C_row[i] += myarg->A_row[j] * *(myarg->B + j * MAT_2 + i);
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    double A[MAT_0][MAT_1], B[MAT_1][MAT_2], C[MAT_0][MAT_2];
    double *a = &(A[0][0]);
    double *b = &(B[0][0]);
    double *c = &(C[0][0]);

    if (myid == 0)
    {
        make_mat(APATH, MAT_0, MAT_1, a);
        make_mat(BPATH, MAT_1, MAT_2, b);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Bcast(&(B[0][0]), MAT_1 * MAT_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        int i, j;
        j = (numprocs - 1) < MAT_0 ? (numprocs - 1) : MAT_0;
        for (i = 1; i < numprocs; i++)
        {
            if (i <= MAT_0)
            {
                MPI_Send(A[i - 1], MAT_1, MPI_DOUBLE, i, 99, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Send(&j, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        int numsend = j;
        for (i = 1; i <= MAT_0; i++)
        {
            int sender = (i - 1) % (numprocs - 1) + 1;
            MPI_Status status;
            MPI_Recv(C[i - 1], MAT_2, MPI_DOUBLE, sender, 100, MPI_COMM_WORLD, &status);
            if (numsend < MAT_0)
            {
                MPI_Send(A[numsend], MAT_1, MPI_DOUBLE, sender, 99, MPI_COMM_WORLD);
                numsend++;
            }
            else
            {
                MPI_Send(&j, 0, MPI_INT, sender, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        int numthreads = get_nprocs();
        pthread_t *tids = (pthread_t *)malloc(numthreads * sizeof(pthread_t));
        struct threadArg *targs = (struct threadArg *)malloc(numthreads * sizeof(struct threadArg));
        int i;
        for (i = 0; i < numthreads; i++)
        {
            targs[i].tid = i;
            targs[i].B = b;
            targs[i].A_row = a;
            targs[i].C_row = c;
            targs[i].numthreads = numthreads;
        }
        while (1)
        {
            MPI_Status status;
            MPI_Recv(a, MAT_1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == 0)
                break;
            for (i = 0; i < numthreads; i++)
            {
                pthread_create(&tids[i], NULL, worker, &targs[i]);
            }
            for (i = 0; i < numthreads; i++)
            {
                pthread_join(tids[i], NULL);
            }
            MPI_Send(c, MAT_2, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (myid == 0)
    {
        printf("Start time:\t %.6lf\n", start);
        printf("End   time:\t %.6lf\n", end);
        printf("Used  time:\t %.6lf\n\n", end - start);
        print_mat(CPATH, MAT_0, MAT_2, c);
        int i, j, k;
        for (i = 0; i < MAT_0; i++)
        {
            for (j = 0; j < MAT_2; j++)
            {
                double tmp = 0;
                for (k = 0; k < MAT_1; k++)
                {
                    tmp += A[i][k] * B[k][j];
                }
                if (tmp != C[i][j])
                {
                    printf("Wrong answer at (%d, %d)\n", i, j);
                }
            }
        }
    }

    MPI_Finalize();

    return 0;
}
