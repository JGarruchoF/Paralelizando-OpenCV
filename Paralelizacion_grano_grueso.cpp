//SPD_P11_plantillaOpenMP.cpp

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

int num_rep = 10;

void procesado_sec();
void procesado_paralelo();
void procesado1(Mat image, Mat& image_dst);
void procesado2(Mat image, Mat& image_dst);
void procesado3(Mat image, Mat& image_dst);
void procesado4(Mat image, Mat& image_dst);

double tiempo_min, tiempo_start, tiempo_end;

Mat image1, image1_dst, image2, image2_dst, image3, image3_dst, image4, image4_dst, image_gray;

int main(int argc, char** argv)
{
    
    const string winname1("Threshold");
    const string winname2("Equalizador");
    const string winname3("Sobel");
    const string winname4("Laplace");
    
    //    tb se puede usar un contructor como:
    //		Mat image = Mat(491,468,CV_8UC1);

    image1 = ourImread("Imagenes/minions_grande.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    image2 = image1.clone();  // copiar la imagen
    image3 = image1.clone();
    image4 = image1.clone();

    if ((!image1.data) || (!image2.data) || (!image3.data) || (!image4.data))                              // Check for invalid input
    {
        cout << "Could not open or find some image" << std::endl;
        cv::waitKey(5000);
        return -1;
    }


	
	
	for (int i = 0; i < num_rep; i++) //Varias iteraciones para reducir error
    {
        tiempo_start = omp_get_wtime();

		procesado_sec();

        tiempo_end = omp_get_wtime();

        double tiempo = (tiempo_end - tiempo_start) * 1000; //en ms

        if (tiempo < tiempo_min)
        {
            tiempo_min = tiempo;
        }



    }
	printf("Tiempo secuencial: %lf ms \n", tiempo_min); // mostrar por consola el tiempo en milisegundos


    for (int i = 0; i < num_rep; i++) //Varias iteraciones para reducir error
    {
        tiempo_start = omp_get_wtime();

        procesado_paralelo();

        tiempo_end = omp_get_wtime();

        double tiempo = (tiempo_end - tiempo_start) * 1000; //en ms

        if (tiempo < tiempo_min)
        {
            tiempo_min = tiempo;
        }



    }
	printf("Tiempo paralelo: %lf ms \n", tiempo_min); // mostrar por consola el tiempo en milisegundos



	namedWindow(  winname1 , CV_WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow(  winname2 , CV_WINDOW_AUTOSIZE );// Create a window for display.
    namedWindow(  winname3 , CV_WINDOW_AUTOSIZE );// Create a window for display.
    namedWindow(  winname4 , CV_WINDOW_AUTOSIZE );// Create a window for display.
	imshow ( winname1 , image1 );                // Show our image inside it.
	imshow ( winname2 , image1_dst );            // Show our image inside it.
    imshow(  winname3 , image3_dst );            // Show our image inside it.
    imshow(  winname4 , image4_dst );            // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}


void procesado_sec()
{
    tiempo_min = 1000000;

    for (int i = 0; i < num_rep; i++)  //Varrias iteraciones para reducir error
    {
        tiempo_start = omp_get_wtime();

        procesado1(image1, image1_dst);
        procesado2(image2, image2_dst);
        procesado3(image3, image3_dst);
        procesado4(image4, image4_dst);

        tiempo_end = omp_get_wtime();

        double tiempo = (tiempo_end - tiempo_start) * 1000; //en ms

        if (tiempo < tiempo_min)
        {
            tiempo_min = tiempo;
        }

    }
    printf("Tiempo sec: %lf ms \n", tiempo_min); // mostrar por consola el tiempo en milisegundos

}

void procesado_paralelo()
{
    omp_set_num_threads(8);

	#pragma omp parallel sections
	{
		#pragma omp  section
		procesado1(image1, image1_dst);
		
	   #pragma omp  section
		procesado2(image2, image2_dst);
		
		#pragma omp  section
		procesado3(image3, image3_dst);
		
		#pragma omp  section
		procesado4(image4, image4_dst);
		
	}


    }

    


}



void procesado1(Mat image, Mat &image_dst) 
{
    Mat image_blur, image_gray;
    GaussianBlur(image, image_blur, Size(5,5), 0);
    cvtColor(image_blur, image_gray, COLOR_RGB2GRAY);
    threshold(image_gray, image_dst, 200, 200, 2);
}

void procesado2(Mat image, Mat &image_dst)
{
    Mat image_blur, image_gray;
    GaussianBlur(image, image_blur, Size(5, 5), 0);
    cvtColor(image_blur, image_gray, COLOR_RGB2GRAY);
    equalizeHist(image_gray, image_dst);
}

void procesado3(Mat image, Mat& image_dst)
{
    Mat image_blur, image_gray, image_threshold, grad_x, abs_grad_x, grad_y, abs_grad_y;
    GaussianBlur(image, image_blur, Size(5, 5), 0);
    cvtColor(image_blur, image_gray, COLOR_RGB2GRAY);
    Sobel(image_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);  
    convertScaleAbs(grad_x, image_dst);
}

void procesado4(Mat image, Mat &image_dst)
{
    Mat image_blur, image_gray, image_lp;
    GaussianBlur(image, image_blur, Size(5, 5), 0);
    cvtColor(image_blur, image_gray, COLOR_RGB2GRAY);
    Laplacian(image_gray, image_lp, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(image_lp, image_dst);
}

