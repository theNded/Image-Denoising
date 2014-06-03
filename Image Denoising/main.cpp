#include "core.h"

using namespace std;
int main(int argc, const char * argv[])
{
    IplImage *src, *in, *out;
    cvNamedWindow("Noisy");
    src = cvLoadImage("/Users/neo/Desktop/lena-20.png");
    in = cvCreateImage(cvGetSize(src), src->depth, 1);
    cvCvtColor(src, in, CV_BGR2GRAY);
    cvShowImage("Noisy", in);


    out = cvBM3D(in, 32, 8, 20);
    //out = cvNLM(in, 33, 7, 2, 20);
    //out = gaussianSmooth(in, 3, 3, 0);
    //out = bilateralSmooth(in);
    //out = wienerSmooth(in, 5, 5);
    cvSaveImage("/Users/neo/Desktop/lena-20-denoised.png", out);
    cvNamedWindow("Denoised");
    cvShowImage("Denoised", out);
    cvWaitKey(-1);
    return 0;
}

