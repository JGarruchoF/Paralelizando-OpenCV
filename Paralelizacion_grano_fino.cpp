
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

#ifdef __cplusplus 
#define ourImread(filename, isColor) cvLoadImage(filename, isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif

int num_rep = 100;

void cambio_brillo_contraste1_sec(Mat image, Mat& new_image, double alpha, int beta);
void cambio_brillo_contraste1_parallel(Mat image, Mat& new_image, double alpha, int beta);
void cambio_brillo_contraste2_sec(Mat image, Mat new_image, double alpha, int beta);
void cambio_brillo_contraste2_parallel(Mat image, Mat new_image, double alpha, int beta);


double tiempo_min_sec, tiempo_min_par, tiempo_start, tiempo_end;

Mat image1;
Mat image1_dst;
Mat image2;
Mat image2_dst;


int main(int argc, char** argv)
{

    const string winname1("Original");
    const string winname2("Modificada1");        
    const string winname3("Modificada2");

  

    image1 = ourImread("Imagenes/minions.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    image1_dst = image1.clone();
    image2 = image1.clone();
    image2_dst = Mat::zeros(image1.size(), image1.type());


    if ((!image1.data))                            // Check for invalid input
    {
        cout << "Could not open or find some image" << std::endl;
        cv::waitKey(5000);
        return -1;
    }


    tiempo_min_sec = 1000000;
    tiempo_min_par = 1000000;

    for (int i = 0; i < num_rep; i++)
    {



        double tiempo;

        tiempo_start = omp_get_wtime();

        cambio_brillo_contraste1_sec(image1, image1_dst, 2.0, 50);
        //cambio_brillo_contraste2_sec(image2, image2_dst, 2.0, 50);

        tiempo_end = omp_get_wtime();

        tiempo = (tiempo_end - tiempo_start) * 1000; //en ms

        if (tiempo < tiempo_min_sec)
        {
            tiempo_min_sec = tiempo;
        }

        tiempo_start = omp_get_wtime();

        cambio_brillo_contraste1_parallel(image2, image1_dst, 2.0, 50);
        //cambio_brillo_contraste2_parallel(image2, image1_dst, 2.0, 50);

        tiempo_end = omp_get_wtime();

        tiempo = (tiempo_end - tiempo_start) * 1000;

        if (tiempo < tiempo_min_par)
        {
            tiempo_min_par = tiempo;
        }

    }
    printf("Tiempo sec: %lf ms \n", tiempo_min_sec); // mostrar por consola el tiempo en milisegundos
    printf("Tiempo par: %lf ms \n", tiempo_min_par); // mostrar por consola el tiempo en milisegundos


#pragma omp barrier

   // printf("Tiempo: %lf ms \n", tiempo_min); // mostrar por consola el tiempo en milisegundos


    namedWindow("Original", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("Modificada1", CV_WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("Modificada2", CV_WINDOW_AUTOSIZE);// Create a window for display.

    imshow("Original", image1);                // Show our image inside it.
    imshow("Modificada1", image1_dst);                   // Show our image inside it.
    imshow("Modificada2", image2_dst);                   // Show our image inside it.



    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}


void cambio_brillo_contraste1_sec(Mat image, Mat& new_image, double alpha, int beta)
{ 
    int filas = image.rows;
    int columnas = image.cols;
    for (int y = 0; y < filas; y++) {
        for (int x = 0; x < columnas; x++) {
            for (int c = 0; c < 3; c++) {
                new_image.at<Vec3b>(y, x)[c] =
                    saturate_cast<uchar>(alpha * (image.at<Vec3b>(y, x)[c]) + beta);
            }
        }
    }

}
void cambio_brillo_contraste1_parallel(Mat image, Mat& new_image, double alpha, int beta)
{

    int filas = image.rows;
    int columnas = image.cols;

    omp_set_num_threads(4);

#pragma omp parallel for default(none) shared (filas, columnas, new_image, image) 
    for (int y = 0; y < filas; y++) {
        for (int x = 0; x < columnas; x++) {
            for (int c = 0; c < 3; c++) {
                new_image.at<Vec3b>(y, x)[c] =
                    saturate_cast<uchar>(alpha * (image.at<Vec3b>(y, x)[c]) + beta);
            }
        }
    }

}

void cambio_brillo_contraste2_sec(Mat image, Mat new_image, double alpha, int beta)
{

    uchar* myData1 = image.data;
    uchar* myData2 = new_image.data;
    int filas = image.rows;
    int columnas = image.cols;
    int stride = image.step;

    for (int y = 0; y < filas; y++) {
        uchar* p1 = &(myData1[y * stride]);
        uchar* p2 = &(myData2[y * stride]);

        for (int x = 0; x < columnas; x++) {
            for (int c = 0; c < 3; c++) {
                int value = alpha * (*p1) + beta;
                *p2 = value > 255 ? 255 : value;

                p1++;
                p2++;
            }
        }
    }

}
void cambio_brillo_contraste2_parallel(Mat image, Mat new_image, double alpha, int beta)
{

    uchar* myData1 = image.data;
    uchar* myData2 = new_image.data;

    int stride = image.step;
    int filas = image.rows;
    int columnas = image.cols;

    omp_set_num_threads(4);

#pragma omp parallel for  default(none) shared (myData2, myData1, filas, columnas, stride)
    for (int y = 0; y < filas; y++) {
        uchar* p1 = &(myData1[y * stride]);
        uchar* p2 = &(myData2[y * stride]);
        for (int x = 0; x < columnas; x++) {
            for (int c = 0; c < 3; c++) {
                int value = alpha * (*p1) + beta;
                *p2 = value > 255 ? 255 : value;

                p1++;
                p2++;
            }
        }
    }

}
