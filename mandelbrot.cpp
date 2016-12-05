/*

COMP90025 Project 1C: MPI and Mandelbrot Set
San Kho Lin (829463) sanl1@student.unimelb.edu.au

Compile with:
    -DSTATIC  - Static Task Assignment
    none      - Dynamic Task Assignment/Work Pool/Processor Farms - This is default

Tested with Intel Compiler on Windows:
    mpicxx.bat mandelbrot.cpp

Tested on VLSCI Snowy:
Please use GCC/g++ to get the consistent output as sequential/serial program

    module unload OpenMPI/1.10.0-iccifort-2015.2.164-GCC-4.9.2
    module load OpenMPI/1.10.0-GCC-4.9.2
    which mpic++
    mpic++ mandelbrot.cpp -o mandelbrot.exe

*/

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#define DATA_TAG    1
#define RESULT_TAG    2
#define TERMINATE_TAG    3

int nprocs;
int rank;

void findmyrange(int n, int nth, int me, int *myrange) {
    int chunksize = n / nth;
    myrange[0] = (me - 1) * chunksize;
    if (me < nth) {
        myrange[1] = (me)* chunksize - 1;
    }
    else {
        myrange[1] = n;
    }
}

void findmyrange0(int n, int nth, int me, int *myrange) {
    int chunksize = n / nth;
    myrange[0] = me * chunksize;
    if (me < nth - 1) {
        myrange[1] = (me + 1) * chunksize - 1;
    }
    else {
        myrange[1] = n - 1;
    }
}

/*
Mandelbrot Set - Sequential Optimizations based on:
    Cardioid / bulb checking optimization
    Escapse Time Algorithm with Periodicity checking

References:
https://en.wikipedia.org/wiki/Mandelbrot_set#Optimizations
https://en.wikipedia.org/wiki/User:Simpsons_contributor/periodicity_checking
http://locklessinc.com/articles/mandelbrot/
*/
int inset(double real, double img, int maxiter) {

    //-- Cardioid / bulb checking optimization

    double yy = img * img;

    // inside cardioid
    double cb_tmp = real - 0.25;
    double q = cb_tmp * cb_tmp + yy;
    double a = q * (q + cb_tmp);
    double b = 0.25 * yy;
    if (a < b) return 1;

    // inside period-2 bulb
    cb_tmp = real + 1.0;
    if ((cb_tmp * cb_tmp) + yy < 0.0625) return 1;

    //-- Escapse Time Algorithm with Periodicity checking

    double z_real = real;
    double z_img = img;
    double tmp, ckr, cki;
    unsigned p = 0, ptot = 8;

    do {
        ckr = z_real;
        cki = z_img;

        ptot += ptot;
        if (ptot > maxiter) ptot = maxiter;

        for (; p < ptot; p++) {

            tmp = z_real*z_real - z_img*z_img + real;
            z_img *= 2.0*z_real;
            z_img += img;
            z_real = tmp;

            if (z_real*z_real + z_img*z_img > 4.0) return 0;

            if ((z_real == ckr) && (z_img == cki)) return 1;
        }

    } while (ptot != maxiter);

    return 1;
}

// return 1 if in set, 0 otherwise
int inset0(double real, double img, int maxiter) {
    double z_real = real;
    double z_img = img;
    for (int iters = 0; iters < maxiter; iters++) {
        double z2_real = z_real*z_real - z_img*z_img;
        double z2_img = 2.0*z_real*z_img;
        z_real = z2_real + real;
        z_img = z2_img + img;
        if (real == z_real && img == z_img) return 1;
        if (z_real*z_real + z_img*z_img > 4.0) return 0;
    }
    return 1;
}

int calc_tag_factor(int region) {
    if (region < 1) return 0;
    return region + 3;
}

void do_dyn_task_master(int argc, char *argv[]) {

    nprocs = nprocs - 1; // 1 less for master
    //printf("nprocs: %d\n", nprocs); //debug 1

    double real_lower;
    double real_upper;
    double img_lower;
    double img_upper;
    int num;
    int maxiter;
    int num_regions = (argc - 1) / 6;

    MPI_Bcast(&num_regions, 1, MPI_INT, 0, MPI_COMM_WORLD); //debug 2

    for (int region = 0; region < num_regions; region++) {
        // scan the arguments
        sscanf(argv[region * 6 + 1], "%lf", &real_lower);
        sscanf(argv[region * 6 + 2], "%lf", &real_upper);
        sscanf(argv[region * 6 + 3], "%lf", &img_lower);
        sscanf(argv[region * 6 + 4], "%lf", &img_upper);
        sscanf(argv[region * 6 + 5], "%i", &num);
        sscanf(argv[region * 6 + 6], "%i", &maxiter);

        //--

        double real_step = (real_upper - real_lower) / num;
        double img_step = (img_upper - img_lower) / num;

        //debug 3
        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&region, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&real_step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&img_step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&real_lower, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&img_lower, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //--

        int tag_factor = calc_tag_factor(region);
        int data_tag = DATA_TAG + tag_factor;
        int result_tag = RESULT_TAG + tag_factor;
        int terminate_tag = TERMINATE_TAG + tag_factor;

        MPI_Status status;
        int id;
        int count = 0, cnt = 0, task_count = 0, row = 0;

        //debug 4
        for (int p = 0; p < nprocs; p++) {
            MPI_Send(&row, 1, MPI_INT, p + 1, data_tag, MPI_COMM_WORLD);
            //printf("%d\n", p);
            task_count++;
            row++;
        }

        //--

        do {
            MPI_Recv(&count, 1, MPI_INT, MPI_ANY_SOURCE, result_tag, MPI_COMM_WORLD, &status);
            task_count--;
            id = status.MPI_SOURCE;

            if (row <= num) {
                //printf("rank: %d request more...\n", id);
                MPI_Send(&row, 1, MPI_INT, id, data_tag, MPI_COMM_WORLD);
                task_count++;
                row++;
            }
            else {
                MPI_Send(&row, 1, MPI_INT, id, terminate_tag, MPI_COMM_WORLD);
            }

            cnt += count;
            //printf("\trank: %d, count: %d\n", id, count);

        } while (task_count > 0);

        printf("%d\n", cnt);
    }
}

void do_dyn_task_slave() {

    int num_regions;
    MPI_Bcast(&num_regions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("rank: %d, num_regions: %d\n", rank, num_regions); //debug 2

    for (int region_cnt = 0; region_cnt < num_regions; region_cnt++) {

        int region, num, maxiter;
        double real_step, img_step, real_lower, img_lower;

        MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&region, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&real_step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&img_step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&real_lower, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&img_lower, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //debug 3
        //printf("rank: %d, num: %d, maxiter: %d, region: %d, real_step: %lf, img_step: %lf, real_lower: %lf, img_lower: %lf\n", 
        //    rank, num, maxiter, region, real_step, img_step, real_lower, img_lower);

        //--

        int tag_factor = calc_tag_factor(region);
        int data_tag = DATA_TAG + tag_factor;
        int result_tag = RESULT_TAG + tag_factor;

        MPI_Status status;
        int row;

        MPI_Recv(&row, 1, MPI_INT, 0, data_tag, MPI_COMM_WORLD, &status);

        //debug 4
        //printf("rank: %d, region: %d, row: %d, tag: %d\n", rank, region, row, status.MPI_TAG);

        //--

        while (status.MPI_TAG == data_tag) {
            int count = 0;
            double cimg = img_lower + row*img_step;
            for (int real = 0; real <= num; real++) {
                count += inset(real_lower + real*real_step, cimg, maxiter);
            }

            MPI_Send(&count, 1, MPI_INT, 0, result_tag, MPI_COMM_WORLD);

            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        };
    }
}

/*
Possibly it could improve the load-balancing 
by randamizing the coordinate points.
*/
void do_static_task(int argc, char *argv[]) {

    double real_lower;
    double real_upper;
    double img_lower;
    double img_upper;
    int num;
    int maxiter;
    int num_regions = (argc - 1) / 6;

    for (int region = 0; region < num_regions; region++) {
        // scan the arguments
        sscanf(argv[region * 6 + 1], "%lf", &real_lower);
        sscanf(argv[region * 6 + 2], "%lf", &real_upper);
        sscanf(argv[region * 6 + 3], "%lf", &img_lower);
        sscanf(argv[region * 6 + 4], "%lf", &img_upper);
        sscanf(argv[region * 6 + 5], "%i", &num);
        sscanf(argv[region * 6 + 6], "%i", &maxiter);

        //--

        int count = 0;
        int myrange[2];

        findmyrange0(num, nprocs, rank, myrange);
        //printf("myrank: %d and myrange: %d to %d\n", rank, myrange[0], myrange[1]);

        double real_step = (real_upper - real_lower) / num;
        double img_step = (img_upper - img_lower) / num;

        for (int real = myrange[0]; real <= myrange[1]; real++) {
            for (int img = 0; img <= num; img++) {
                count += inset(real_lower + real*real_step, img_lower + img*img_step, maxiter);
            }
        }

        //printf("myrank: %d myrange: %d to %d mycount: %d\n", rank, myrange[0], myrange[1], count);

        int global_count;
        MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%d\n", global_count);
        }
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2) {
        fprintf(stderr, "Number of processes must be at least 2\n");
        MPI_Finalize(); exit(EXIT_FAILURE);
    }

#ifdef STATIC

    //double start_time;
    //if (rank == 0) start_time = MPI_Wtime();

    do_static_task(argc, argv);

    //if (rank == 0) printf("\nElapsed time(second): %f\n", (float)(MPI_Wtime() - start_time));

#else

    if (rank == 0) {

        //double start_time = MPI_Wtime();

        do_dyn_task_master(argc, argv);

        //printf("\nElapsed time(second): %f\n", (float)(MPI_Wtime() - start_time));

    }
    else {
        // must be in else block
        do_dyn_task_slave();
    }

#endif

    MPI_Finalize();

    return EXIT_SUCCESS;
}
