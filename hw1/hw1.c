#include <stdio.h>
#include <mpi.h>
#define ROUDN_TIME 100000                                               //设置循环次数

int main( int argc, char *argv[] )
{
    //存放进程有关信息
    int myid, numprocs;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init( &argc, &argv );                                           //开始多进程
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );                             //获得进程号
    MPI_Comm_size( MPI_COMM_WORLD, &numprocs );                         //返回通讯子进程数
    MPI_Get_processor_name(processor_name, &namelen);                   //获取设备名称
    
    //计算本进程接收信息来源与发送信息的接收方
    int target_pid = (myid + 1) % numprocs;                             //发送信息的接收方
    int source_pid = myid - 1;                                          //接收信息的发送方
    if (source_pid < 0)
    {
        source_pid = numprocs - 1;
    }
    
    int i;
    int received;

    //使用进程0进行时间统计
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);                                        //使用阻塞同步各进程
    if (myid == 0)                                                      //获取时间
    {
        start = MPI_Wtime();
    }

    //开始主要循环
    for (i = 0; i < ROUDN_TIME; i++)
    {
        //进程0首先发送消息，开始循环，并接收最后一个消息
        if (myid == 0)
        {
            /*阻塞地发送信息，函数参数为：
                myid                发送信息内容，指定为发送方进程号
                1                   发送消息数量
                MPI_INT             发送数据类型
                target_pid          接收方进程号
                i                   消息标签，指定为当前发送论次数
                MPI_COMM_WORLD      消息域
            */
            MPI_Send(&myid, 1, MPI_INT, target_pid, i, MPI_COMM_WORLD); 
            /*阻塞地接收信息，函数参数为：
                received            接收信息的内存地址
                1                   接收消息数量
                MPI_INT             接收数据类型
                target_pid          发送方进程号
                i                   消息标签，指定为当前发送论次数
                MPI_COMM_WORLD      消息域
                MPI_STATUS_IGNORE   接收状态，设置为忽略
            */
            MPI_Recv(&received, 1, MPI_INT, source_pid, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //其他进程收到消息后发送新的消息
        else
        {
            //收取消息，同前设置
            MPI_Recv(&received, 1, MPI_INT, source_pid, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //发送消息至下一进程，同前设置
            MPI_Send(&myid, 1, MPI_INT, target_pid, i, MPI_COMM_WORLD);
        }  
    }

    //使用进程0统计时间
    MPI_Barrier(MPI_COMM_WORLD);                                        //使用阻塞同步各进程
    if (myid == 0)
    {
        end = MPI_Wtime();
        printf("Start Time: %lf\n", start);
        printf("End   Time: %lf\n", end);
        printf("All   Time: %lf\n", end - start);
        printf("Round Time: %d\n", ROUDN_TIME);
        printf("Once  Time: %lf\n", (end - start) / ROUDN_TIME);
    }

    MPI_Finalize();                                                     //结束进程
    return 0;
}