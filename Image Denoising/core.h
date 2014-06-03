//
//  Created by Neo on 14-5-28.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#ifndef testCV_core_h
#define testCV_core_h

#include <iostream>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/highgui.h>

static IplImage *out;

static void printMat(CvMat *weigh) {
    for (int ii = 0; ii < weigh->rows; ii ++) {
        for (int jj = 0; jj < weigh->cols; jj ++) {
            double test = *(double *)(weigh->data.ptr + weigh->step * ii + sizeof(double) * jj);
            std::cout << test << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "<<<<<<<<<<<<<<>>>>>>>>>>>>>>>" << std::endl;
}

IplImage * gaussianSmooth(IplImage *, int, int, double);
IplImage * bilateralSmooth(IplImage *);
IplImage * wienerSmooth(IplImage *, int, int);
IplImage * cvNLM(IplImage *in, int windowCalc, int windowSSD, double sigmas, double sigmar);
IplImage * cvBM3D(IplImage *in, int windowSearch, int windowBlock, double sigma);

#endif
