//
//  cvGaussianSmooth.cpp
//  testCV
//
//  Created by Neo on 14-5-29.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#include "core.h"

IplImage * gaussianSmooth(IplImage *in, int windowX, int windowY, double sigma = 0) {
    out = cvCreateImage(cvSize(in->width, in->height), in->depth, in->nChannels);
    cvSmooth(in, out, CV_GAUSSIAN, windowX, windowY);
    return out;
}