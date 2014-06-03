//
//  cvBilateralSmooth.cpp
//  testCV
//
//  Created by Neo on 14-5-29.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#include "core.h"

IplImage * bilateralSmooth(IplImage *in) {
    out = cvCreateImage(cvSize(in->width, in->height), in->depth, in->nChannels);
    cvSmooth(in, out, CV_BILATERAL, 0, 0, 35, 10);
    return out;
}
