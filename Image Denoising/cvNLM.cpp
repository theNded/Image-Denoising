//
//  cvNLM.cpp
//  testCV
//
//  Created by Neo on 14-5-29.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#include "core.h"

IplImage * cvNLM(IplImage *in, int windowCalc, int windowSSD, double sigmas, double sigmar) {
    assert((windowCalc & 1) && (windowSSD & 1));
    IplImage *padding = cvCreateImage(cvSize(in->width + windowCalc + windowSSD,
                                             in->height + windowCalc+ windowSSD),
                                      in->depth,
                                      in->nChannels);
    out = cvCreateImage(cvSize(in->width, in->height), in->depth, in->nChannels);
    
    CvPoint offset = cvPoint((windowCalc + windowSSD) >> 1, (windowCalc + windowSSD) >> 1);
    cvCopyMakeBorder(in, padding, offset, IPL_BORDER_REPLICATE);
    
    int windowCalc_2 = windowCalc >> 1, windowSSD_2 = windowSSD >> 1;
    
    CvMat
    *ctr = cvCreateMat(windowSSD, windowSSD, CV_MAKETYPE(CV_32F, in->nChannels)),
    *cmp = cvCreateMat(windowSSD, windowSSD, CV_MAKETYPE(CV_32F, in->nChannels)),
    *gaussian = cvCreateMat(windowSSD, windowSSD, CV_MAKETYPE(CV_32F, in->nChannels)),

    *weigh=cvCreateMat(windowCalc, windowCalc, CV_MAKETYPE(CV_32F, in->nChannels)),
    *val = cvCreateMat(windowCalc, windowCalc, CV_MAKETYPE(CV_32F, in->nChannels)),
    
    *bufSmall = cvCreateMat(windowSSD, windowSSD, CV_MAKETYPE(CV_32F, in->nChannels)),
    *bufLarge = cvCreateMat(windowCalc, windowCalc, CV_MAKETYPE(CV_32F, in->nChannels));
    
    CvScalar ssdScalar, ttlScalar;
    float *ptrWeigh = weigh->data.fl, *ptrF;
    uchar *ptrOut = (uchar *)out->imageData, *ptrU;
    
    /* Calculate Gaussian Kernel at first */
    for (int i = - windowSSD_2; i <= windowSSD_2; i ++) {
        for (int j = - windowSSD_2; j <= windowSSD_2; j ++) {
            ptrF = gaussian->data.fl +
            (j + windowSSD_2) * gaussian->step / sizeof(float) +
            (i + windowSSD_2) * in->nChannels;
            for (int k = 0; k < in->nChannels; k ++) {
                *ptrF = i * i + j * j;
                ptrF ++;
            }
        }
    }
    printMat(gaussian);
    ssdScalar = cvScalar(-0.5 / (sigmas * sigmas));
    cvSet(bufSmall, ssdScalar);
    cvMul(gaussian, bufSmall, gaussian);
    cvExp(gaussian, gaussian);
    ssdScalar = cvSum(gaussian);
    cvSet(bufSmall, ssdScalar);
    cvDiv(gaussian, bufSmall, gaussian);
    printMat(gaussian);
    
    for (int i = 0; i < in->width; i ++) {
        for (int j = 0; j < in->height; j ++) {
            /* Get Pixel Matrix For Calculation (Multiply by weight matrix) */
            cvSetImageROI(padding, cvRect(offset.x + i - windowCalc_2,
                                          offset.y + j - windowCalc_2,
                                          windowCalc,
                                          windowCalc));
            cvConvert(padding, val);
            cvResetImageROI(padding);
            
            /* Get Center SSD Matrix for Efficiency */
            cvSetImageROI(padding, cvRect(offset.x + i - windowSSD_2,
                                          offset.y + j - windowSSD_2,
                                          windowSSD,
                                          windowSSD));
            cvConvert(padding, ctr);
            cvResetImageROI(padding);
            
            /* Iterate & Calculate each SSD */
            for (int k = - windowCalc_2; k <= windowCalc_2; k ++) {
                for (int l = - windowCalc_2; l <= windowCalc_2; l ++) {
                    /* Get Near SSD Matrix for Calculation */
                    cvSetImageROI(padding,
                                  cvRect(offset.x + i + k - windowSSD_2,
                                         offset.y + j + l - windowSSD_2,
                                         windowSSD,
                                         windowSSD));
                    cvConvert(padding, cmp);
                    cvResetImageROI(padding);
                    
                    /* Calculate SSD */
                    cvSub(ctr, cmp, bufSmall);
                    cvMul(bufSmall, bufSmall, cmp);
                    cvMul(cmp, gaussian, cmp);
                    ssdScalar = cvSum(cmp);
                    
                    /* Write back to the weight matrix */
                    ptrF = ptrWeigh +
                    (l + windowCalc_2) * weigh->step / sizeof(float) +
                    (k + windowCalc_2) * in->nChannels;
                    for (int m = 0; m < in->nChannels; m ++) {
                        *ptrF = ssdScalar.val[m];
                        ptrF ++;
                    }
                }
            }
            cvSet(bufLarge, cvScalar(-0.5 / (sigmar * sigmar)));
            cvMul(weigh, bufLarge, weigh);
            cvExp(weigh, weigh);
            
            /* Normalize */
            ttlScalar = cvSum(weigh);
            cvSet(bufLarge, ttlScalar);
            cvDiv(weigh, bufLarge, weigh);

            /* Weighed Pixels */
            cvMul(weigh, val, val);
            ttlScalar = cvSum(val);
            ptrU = ptrOut + j * out->widthStep + i * in->nChannels;
            for (int m = 0; m < in->nChannels; m ++) {
                *ptrU = (uchar) ttlScalar.val[m];
                ptrU ++;
            }
        }
    }
    return out;
}