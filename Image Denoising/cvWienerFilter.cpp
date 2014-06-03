//
//  cvWienerFilter.cpp
//  testCV
//
//  Created by Neo on 14-5-29.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#include "core.h"

IplImage * wienerSmooth(IplImage *in, int windowX, int windowY) {
    out = cvCreateImage(cvSize(in->width, in->height), in->depth, in->nChannels);
    
    CvMat *p_kernel = NULL;
    CvMat *srcMat = NULL, *dstMat = NULL;
    CvMat *p_tmpMat1, *p_tmpMat2, *p_tmpMat3, *p_tmpMat4;
	double noise_power;
    
    p_kernel = cvCreateMat(windowX, windowY, CV_64F);
	cvSet(p_kernel, cvScalar( 1.0 / (double) (windowX * windowY)));
    
	//Convert to matrices
    srcMat = cvCreateMat(in->width, in->height, CV_8U);
    dstMat = cvCreateMat(in->width, in->height, CV_8U);
    cvConvert(in, srcMat);
    
	// Now create a temporary holding matrix
	p_tmpMat1 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat2 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat3 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat4 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
    
	// Local mean of input
	cvFilter2D(srcMat, p_tmpMat1, p_kernel);
    
	//Local variance of input
	cvMul(srcMat, srcMat, p_tmpMat2);
	cvFilter2D(p_tmpMat2, p_tmpMat3, p_kernel);
    
	//Subtract off local_mean^2 from local variance
	cvMul(p_tmpMat1, p_tmpMat1, p_tmpMat4); //localMean^2
	cvSub(p_tmpMat3, p_tmpMat4, p_tmpMat3); //filter(in^2) - localMean^2 ==> localVariance
    
	//Estimate noise power
	noise_power = cvMean(p_tmpMat3, 0);
    
	cvSub (srcMat, p_tmpMat1, dstMat);		     //in - local_mean
	cvMaxS(p_tmpMat3, noise_power, p_tmpMat2); //max(localVar, noise)
    
	cvAddS(p_tmpMat3, cvScalar(-noise_power), p_tmpMat3); //localVar - noise
	cvMaxS(p_tmpMat3, 0, p_tmpMat3); // max(0, localVar - noise)
    
	cvDiv (p_tmpMat3, p_tmpMat2, p_tmpMat3);  //max(0, localVar-noise) / max(localVar, noise)
    
	cvMul (p_tmpMat3, dstMat, dstMat);
	cvAdd (dstMat, p_tmpMat1, dstMat);
    
    out = cvGetImage(dstMat, out);
    
	cvReleaseMat(&p_kernel );
    cvReleaseMat(&srcMat);
    cvReleaseMat(&dstMat);
	cvReleaseMat(&p_tmpMat1);
	cvReleaseMat(&p_tmpMat2);
	cvReleaseMat(&p_tmpMat3);
	cvReleaseMat(&p_tmpMat4);
    return out;
}
