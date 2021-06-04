#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <pthread.h>

#define APATH "./mats/A.mat"
#define BPATH "./mats/B.mat"
#define CPATH "./mats/C.mat"

struct threadArg
{
    int tid;
    double *buffB;
    double *buffA;
    double *buffC;
    int dim[3];
    int numthreads;
};

void *worker(void *arg)
{
    int i, j, k;
    struct threadArg *myarg = (struct threadArg *)arg;
    for (i = myarg->tid; i < myarg->dim[0]; i += myarg->numthreads)
    {
        for (j = 0; j < myarg->dim[2]; j++)
        {
            for (k = 0; k < myarg->dim[1]; k++)
            {
                myarg->buffC[i * myarg->dim[2] + j] +=
                    myarg->buffA[i * myarg->dim[1] + k] * myarg->buffB[k * myarg->dim[2] + j];
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    double *A, *B, *C;
    double *buffA, *buffB, *buffC;
    int dim[3];

    int rp = (int)sqrt(numprocs);
    if (myid >= rp * rp)
    {
        printf("Processor number should be a square, kill process %d\n", myid);
        MPI_Finalize();
        exit(0);
    }
    printf("Alive1 %d\n", myid);
    if (myid == 0)
    {
        FILE *matA, *matB;
        if ((!(matA = fopen(APATH, "r"))) || (!(matB = fopen(BPATH, "r"))))
        {
            printf("Can't open file for mat A or B\n");
            MPI_Finalize();
            exit(-1);
        }
        struct stat fstat;
        int fsize;
        char *streamA, *streamB;

        stat(APATH, &fstat);
        fsize = fstat.st_size;
        streamA = (char *)malloc(fsize);
        fread(streamA, sizeof(char), fsize, matA);
        stat(BPATH, &fstat);
        fsize = fstat.st_size;
        streamB = (char *)malloc(fsize);
        fread(streamB, sizeof(char), fsize, matB);

        dim[0] = ((int *)streamA)[0];
        dim[1] = ((int *)streamA)[1];
        dim[2] = ((int *)streamB)[1];

        if (dim[1] != ((int *)streamB)[0])
        {
            printf("Mat A-D1 != B-D0\n");
            MPI_Finalize();
            exit(-2);
        }

        A = (double *)(streamA + sizeof(int) * 2);
        B = (double *)(streamB + sizeof(int) * 2);
        C = (double *)(malloc(sizeof(double) * dim[0] * dim[2]));
    }
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int maxrows_a = (dim[0] + rp - 1) / rp;
    int maxcols_a = (dim[1] + rp - 1) / rp;
    int maxrows_b = maxcols_a;
    int maxcols_b = (dim[2] + rp - 1) / rp;
    printf("Alive2 %d\n", myid);

    int buff_sizeA = sizeof(double) * maxrows_a * maxcols_a;
    int buff_sizeB = sizeof(double) * maxrows_b * maxcols_b;
    int buff_sizeC = sizeof(double) * maxrows_a * maxcols_b;
    buffA = (double *)malloc(buff_sizeA);
    buffB = (double *)malloc(buff_sizeB);
    buffC = (double *)malloc(buff_sizeC);
    double *recvA = (double *)malloc(buff_sizeA);
    double *recvB = (double *)malloc(buff_sizeB);
    memset(buffC, 0, buff_sizeC);
    printf("Alive3 %d\n", myid);
    if (myid == 0)
    {
        int i, j;
        MPI_Request req[rp * rp * 2];
        double **send_tmp = (double **)malloc(sizeof(double *) * rp * rp * 2);
        int req_count = 0;
        for (i = 0; i < rp; i++)
        {
            for (j = 0; j < rp; j++)
            {
                int a_j = j - i;
                if (a_j < 0)
                    a_j = rp + a_j;
                int target = i * rp + a_j;
                send_tmp[req_count] = (double *)malloc(buff_sizeA);
                int ii, jj, count;
                count = 0;
                for (ii = 0; ii < maxrows_a; ii++)
                {
                    for (jj = 0; jj < maxcols_a; jj++)
                    {

                        double this_one = 0;
                        if (i * maxrows_a + ii < dim[0] && j * maxcols_a + jj < dim[1])
                        {
                            this_one = A[(i * maxrows_a + ii) * dim[1] + j * maxcols_a + jj];
                        }
                        if (target == 0)
                        {
                            buffA[count] = this_one;
                        }
                        else
                        {
                            send_tmp[req_count][count] = this_one;
                        }
                        count++;
                    }
                }
                if (target != 0)
                {
                    MPI_Isend(send_tmp[req_count], buff_sizeA / sizeof(double), MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &(req[req_count]));
                    req_count++;
                }
                int b_i = i - j;
                if (b_i < 0)
                    b_i = rp + b_i;
                target = b_i * rp + j;
                send_tmp[req_count] = (double *)malloc(buff_sizeB);
                count = 0;
                for (ii = 0; ii < maxrows_b; ii++)
                {
                    for (jj = 0; jj < maxcols_b; jj++)
                    {
                        double this_one = 0;
                        if (i * maxrows_b + ii < dim[1] && j * maxcols_b + jj < dim[2])
                        {
                            this_one = B[(i * maxrows_b + ii) * dim[2] + j * maxcols_b + jj];
                        }
                        if (target == 0)
                        {
                            buffB[count] = this_one;
                        }
                        else
                        {
                            send_tmp[req_count][count] = this_one;
                        }
                        count++;
                    }
                }
                if (target != 0)
                {
                    MPI_Isend(send_tmp[req_count], buff_sizeB / sizeof(double), MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &(req[req_count]));
                    req_count++;
                }
            }
        }
        MPI_Waitall(req_count, req, MPI_STATUS_IGNORE);
        for (i = 0; i < req_count; i++)
        {
            free(send_tmp[i]);
        }
        free(send_tmp);
    }
    else
    {
        MPI_Request req[2];
        MPI_Irecv(buffA, buff_sizeA / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, req);
        MPI_Irecv(buffB, buff_sizeB / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &(req[1]));
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    }
    printf("Alive4 %d\n", myid);

    int self_i = myid / rp;
    int self_j = myid % rp;
    int sa = self_j - 1;
    if (sa < 0)
        sa += rp;
    sa += self_i * rp;
    int sb = self_i - 1;
    if (sb < 0)
        sb += rp;
    sb = sb * rp + self_j;
    int ra = self_j + 1;
    if (ra >= rp)
        ra -= rp;
    ra += self_i * rp;
    int rb = self_i + 1;
    if (rb >= rp)
        rb -= rp;
    rb = rb * rp + self_j;

    printf("Alive5 %d\n", myid);
    int count = 0;

    printf("Alive6 %d\n", myid);
    int numthreads = get_nprocs();
    pthread_t *tids = (pthread_t *)malloc(numthreads * sizeof(pthread_t));
    struct threadArg *targs = (struct threadArg *)malloc(numthreads * sizeof(struct threadArg));
    int i;
    for (i = 0; i < numthreads; i++)
    {
        targs[i].tid = i;
        targs[i].buffB = buffB;
        targs[i].buffA = buffA;
        targs[i].buffC = buffC;
        targs[i].dim[0] = maxrows_a;
        targs[i].dim[1] = maxrows_b;
        targs[i].dim[2] = maxcols_b;
        targs[i].numthreads = numthreads;
    }
    while (1)
    {
        count++;
        printf("Begin compute at %d, time %d\n", myid, count);
        // int i, j, k;
        // for (i = 0; i < maxrows_a; i++)
        // {
        //     for (j = 0; j < maxcols_b; j++)
        //     {
        //         for (k = 0; k < maxcols_a; k++)
        //         {
        //             buffC[i * maxcols_b + j] += buffA[i * maxcols_a + k] * buffB[k * maxcols_b + j];
        //         }
        //     }
        // }
        int i;
        for (i = 0; i < numthreads; i++)
        {
            pthread_create(&tids[i], NULL, worker, &targs[i]);
        }
        for (i = 0; i < numthreads; i++)
        {
            pthread_join(tids[i], NULL);
        }
        if (count >= rp)
        {
            break;
        }
        printf("Begin transfer at %d, time %d\n", myid, count);
        MPI_Request req[4];
        MPI_Isend(buffA, buff_sizeA / sizeof(double), MPI_DOUBLE, sa, 0, MPI_COMM_WORLD, req);
        MPI_Isend(buffB, buff_sizeB / sizeof(double), MPI_DOUBLE, sb, 0, MPI_COMM_WORLD, &(req[1]));
        MPI_Irecv(recvA, buff_sizeA / sizeof(double), MPI_DOUBLE, ra, 0, MPI_COMM_WORLD, &(req[2]));
        MPI_Irecv(recvB, buff_sizeB / sizeof(double), MPI_DOUBLE, rb, 0, MPI_COMM_WORLD, &(req[3]));
        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
        memcpy(buffA, recvA, buff_sizeA);
        memset(recvA, 0, buff_sizeA);
        memcpy(buffB, recvB, buff_sizeB);
        memset(recvB, 0, buff_sizeB);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Alive7 %d\n", myid);
    if (myid != 0)
    {
        MPI_Send(buffC, buff_sizeC / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        int i, j;
        for (i = 0; i < rp; i++)
        {
            for (j = 0; j < rp; j++)
            {
                int source = i * rp + j;
                if (source != 0)
                {
                    MPI_Recv(buffC, buff_sizeC / sizeof(double), MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                printf("Begin write from %d\n", source);
                int ii, jj;
                count = 0;
                for (ii = 0; ii < maxrows_a; ii++)
                {
                    for (jj = 0; jj < maxcols_b; jj++)
                    {
                        if (i * maxrows_a + ii < dim[0] && j * maxcols_b + jj < dim[2])
                        {
                            C[(i * maxrows_a + ii) * dim[2] + j * maxcols_b + jj] = buffC[count];
                        }
                        count++;
                    }
                }
            }
        }
        FILE *c_file;
        if (!(c_file = fopen(CPATH, "w")))
        {
            printf("Can't open file %s\n", CPATH);
        }
        fwrite(dim, sizeof(int), 1, c_file);
        fwrite(&(dim[2]), sizeof(int), 1, c_file);
        fwrite(C, sizeof(double), dim[0] * dim[2], c_file);
        fclose(c_file);
    }

    MPI_Finalize();
    return 0;
}
