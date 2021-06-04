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

// 计算时开启多线程的辅助结构体
struct threadArg
{
    int tid;                // 线程ID
    double *buffB;          // 子矩阵B
    double *buffA;          // 子矩阵A
    double *buffC;          // 子矩阵C，记录结果
    int dim[3];             // 记录子矩阵大小
    int numthreads;         // 记录线程总数
};

// 计算时开启多线程后每个线程任务
void *worker(void *arg)
{
    int i, j, k;
    struct threadArg *myarg = (struct threadArg *)arg;
    // 循环为每个线程分配子矩阵A的一行，与子矩阵B相乘，更新部分子矩阵C中数据
    // i代表子矩阵A与子矩阵C中的行数
    // j代表子矩阵B与子矩阵C中的列数
    // k代表子矩阵A的列数与子矩阵B的行数
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

// 主函数
int main(int argc, char **argv)
{
    // 初始化各进程，获取当前进程ID与总进程个数
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    // 初始化数据用指针
    double *A, *B, *C;                  // 矩阵A、矩阵B、矩阵C数据，只在主进程使用
    double *buffA, *buffB, *buffC;      // 子矩阵A、B、C，在各进程计算时使用
    int dim[3];                         // 记录矩阵整体的行列数信息
    // 查看进程总数是否为平方数
    int rp = (int)sqrt(numprocs);
    // 将多余的进程结束
    if (myid >= rp * rp)
    {
        printf("Processor number should be a square, kill process %d\n", myid);
        MPI_Finalize();
        exit(0);
    }
    // 主进程从文件读取数据到内存，修改自辅助程序，注释省略
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
        // 记录矩阵维数信息
        dim[0] = ((int *)streamA)[0];
        dim[1] = ((int *)streamA)[1];
        dim[2] = ((int *)streamB)[1];
        if (dim[1] != ((int *)streamB)[0])
        {
            printf("Mat A-D1 != B-D0\n");
            MPI_Finalize();
            exit(-2);
        }
        // 获取矩阵A、B，为矩阵C分配内存
        A = (double *)(streamA + sizeof(int) * 2);
        B = (double *)(streamB + sizeof(int) * 2);
        C = (double *)(malloc(sizeof(double) * dim[0] * dim[2]));
    }
    // 广播矩阵维数信息，计算子矩阵维数
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int maxrows_a = (dim[0] + rp - 1) / rp;         // 子矩阵A、C的行数
    int maxcols_a = (dim[1] + rp - 1) / rp;         // 子矩阵A的列数
    int maxrows_b = maxcols_a;                      // 子矩阵B的行数
    int maxcols_b = (dim[2] + rp - 1) / rp;         // 子矩阵B、C的列数
    // 计算子矩阵占用内存大小
    int buff_sizeA = sizeof(double) * maxrows_a * maxcols_a;    // 子矩阵A占用大小
    int buff_sizeB = sizeof(double) * maxrows_b * maxcols_b;    // 子矩阵B占用大小
    int buff_sizeC = sizeof(double) * maxrows_a * maxcols_b;    // 子矩阵C占用大小
    // 为子矩阵分配内存空间
    buffA = (double *)malloc(buff_sizeA);
    buffB = (double *)malloc(buff_sizeB);
    buffC = (double *)malloc(buff_sizeC);
    // 在进行数据交换时额外对子矩阵A、B进行缓存，在此开辟内存
    double *recvA = (double *)malloc(buff_sizeA);
    double *recvB = (double *)malloc(buff_sizeB);
    // 初始化子矩阵C各项为0
    memset(buffC, 0, buff_sizeC);
    // 主进程分配数据到各进程
    if (myid == 0)
    {
        int i, j;
        // 异步传输，记录每次发送
        MPI_Request req[rp * rp * 2];
        // 为每次发送开辟内存指针
        double **send_tmp = (double **)malloc(sizeof(double *) * rp * rp * 2);
        int req_count = 0;
        for (i = 0; i < rp; i++)
        {
            for (j = 0; j < rp; j++)
            {
                // 传输第i行第j列的子矩阵A与B
                // 计算当前子矩阵A的传输目标进程号
                int a_j = j - i;
                if (a_j < 0)
                    a_j = rp + a_j;
                int target = i * rp + a_j;
                // 为发送数据开辟缓存
                send_tmp[req_count] = (double *)malloc(buff_sizeA);
                // 从矩阵A中取出当前子矩阵
                int ii, jj, count;
                count = 0;
                for (ii = 0; ii < maxrows_a; ii++)
                {
                    for (jj = 0; jj < maxcols_a; jj++)
                    {
                        // 处理子矩阵中第ii行第jj列的数据，数据超出源矩阵范围时置0
                        double this_one = 0;
                        // 未超出时则读取源矩阵数据
                        if (i * maxrows_a + ii < dim[0] && j * maxcols_a + jj < dim[1])
                        {
                            this_one = A[(i * maxrows_a + ii) * dim[1] + j * maxcols_a + jj];
                        }
                        // 当子矩阵将在主进程使用时，直接存储
                        if (target == 0)
                        {
                            buffA[count] = this_one;
                        }
                        // 当子矩阵将被传输到其他进程时，写入缓存
                        else
                        {
                            send_tmp[req_count][count] = this_one;
                        }
                        count++;
                    }
                }
                // 对于不在主进程使用的子矩阵，将其非阻塞地发送到对应进程
                if (target != 0)
                {
                    MPI_Isend(send_tmp[req_count], buff_sizeA / sizeof(double), 
                        MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &(req[req_count]));
                    req_count++;
                }
                // 开始处理第i行第j列的子矩阵B，处理过程同上方对于A的处理过程，注释省略
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
        // 等待所有数据发送完毕
        MPI_Waitall(req_count, req, MPI_STATUS_IGNORE);
        // 释放发送数据用到的缓存
        for (i = 0; i < req_count; i++)
        {
            free(send_tmp[i]);
        }
        free(send_tmp);
    }
    // 其他进程接收从主进程分发来的数据
    else
    {
        MPI_Request req[2];
        // 非阻塞地接收子矩阵A与B
        MPI_Irecv(buffA, buff_sizeA / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, req);
        MPI_Irecv(buffB, buff_sizeB / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &(req[1]));
        // 等待接收完成
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    }
    // 预先计算每次交互数据时，数据的去向以及新数据的来源
    int self_i = myid / rp;         // 当前进程处理的子矩阵在源矩阵的行数
    int self_j = myid % rp;         // 当前进程处理的子矩阵在源矩阵的列数
    int sa = self_j - 1;            // 计算当前进程发送出去的子矩阵A的接收者
    if (sa < 0)
        sa += rp;
    sa += self_i * rp;
    int sb = self_i - 1;            // 计算当前进程发送出去的子矩阵B的接收者
    if (sb < 0)
        sb += rp;
    sb = sb * rp + self_j;
    int ra = self_j + 1;            // 计算当前进程要接收的新子矩阵A的发送者
    if (ra >= rp)
        ra -= rp;
    ra += self_i * rp;
    int rb = self_i + 1;            // 计算当前进程要接收的新子矩阵B的发送者
    if (rb >= rp)
        rb -= rp;
    rb = rb * rp + self_j;

    // 记录交互子矩阵的次数
    int count = 0;
    // 获取当前CPU的核心数，用于开启合适的线程数
    int numthreads = get_nprocs();
    // 为开启线程时传递的参数开设内存
    pthread_t *tids = (pthread_t *)malloc(numthreads * sizeof(pthread_t));
    struct threadArg *targs = (struct threadArg *)malloc(numthreads * sizeof(struct threadArg));
    int i;
    // 为开启线程时传递的参数赋值
    for (i = 0; i < numthreads; i++)
    {
        targs[i].tid = i;                       // 线程号
        targs[i].buffB = buffB;                 // 子矩阵B指针
        targs[i].buffA = buffA;                 // 子矩阵A指针
        targs[i].buffC = buffC;                 // 子矩阵C指针
        targs[i].dim[0] = maxrows_a;            // 子矩阵行列信息
        targs[i].dim[1] = maxrows_b;
        targs[i].dim[2] = maxcols_b;
        targs[i].numthreads = numthreads;       // 线程总数
    }
    // 循环计算并交互数据
    while (1)
    {
        count++;
        printf("Begin compute at %d, time %d\n", myid, count);
        int i;
        // 开启多线程，计算子矩阵A与B相乘结果，更新子矩阵C
        for (i = 0; i < numthreads; i++)
        {
            pthread_create(&tids[i], NULL, worker, &targs[i]);
        }
        // 等待各进程结束
        for (i = 0; i < numthreads; i++)
        {
            pthread_join(tids[i], NULL);
        }
        // 当数据交互次数达到上限，即所有计算完成时，跳出循环
        if (count >= rp)
        {
            break;
        }
        // 进行一次子矩阵的数据交换，子矩阵A进程循环左移，子矩阵B进行循环上移
        // 发送当前子矩阵A与B，并接收新的子矩阵A与B，发送与接收全部采用非阻塞方式
        printf("Begin transfer at %d, time %d\n", myid, count);
        MPI_Request req[4];
        MPI_Isend(buffA, buff_sizeA / sizeof(double), MPI_DOUBLE, sa, 0, MPI_COMM_WORLD, req);
        MPI_Isend(buffB, buff_sizeB / sizeof(double), MPI_DOUBLE, sb, 0, MPI_COMM_WORLD, &(req[1]));
        MPI_Irecv(recvA, buff_sizeA / sizeof(double), MPI_DOUBLE, ra, 0, MPI_COMM_WORLD, &(req[2]));
        MPI_Irecv(recvB, buff_sizeB / sizeof(double), MPI_DOUBLE, rb, 0, MPI_COMM_WORLD, &(req[3]));
        // 等待数据交互全部完成
        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
        // 把新得到的子矩阵A与B从缓存中复制到运算用的内存中
        memcpy(buffA, recvA, buff_sizeA);
        memset(recvA, 0, buff_sizeA);
        memcpy(buffB, recvB, buff_sizeB);
        memset(recvB, 0, buff_sizeB);
    }
    // 计算完成，进行结果交换
    MPI_Barrier(MPI_COMM_WORLD);
    // 非主进程，将计算结果发送给主进程
    if (myid != 0)
    {
        MPI_Send(buffC, buff_sizeC / sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    // 主进程接收计算结果并拼接出矩阵C
    else
    {
        int i, j;
        for (i = 0; i < rp; i++)
        {
            for (j = 0; j < rp; j++)
            {
                // 处理第i行第j列处的子矩阵C
                // 计算该数据来源的进程号
                int source = i * rp + j;
                // 数据不在主进程时，从对应进程接收
                if (source != 0)
                {
                    MPI_Recv(buffC, buff_sizeC / sizeof(double), 
                        MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                printf("Begin write from %d\n", source);
                // 将数据拼接进矩阵C
                int ii, jj;
                count = 0;
                for (ii = 0; ii < maxrows_a; ii++)
                {
                    for (jj = 0; jj < maxcols_b; jj++)
                    {
                        // 子矩阵C中数据可能不在矩阵C内，因为划分子矩阵可能存在不能整除的情况
                        // 数据在矩阵C内时则写入对应位置，否则丢弃
                        if (i * maxrows_a + ii < dim[0] && j * maxcols_b + jj < dim[2])
                        {
                            C[(i * maxrows_a + ii) * dim[2] + j * maxcols_b + jj] = buffC[count];
                        }
                        count++;
                    }
                }
            }
        }
        // 将矩阵C计算结果写入文件，程序修改自辅助程序，注释略
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
    // 结束各进程，结束程序
    MPI_Finalize();
    return 0;
}
