/**
 * File: FSUPER.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 * Distance function has been modified
 *
 */


#include <vector>
#include <string>
#include <sstream>
#include <stdint-gcc.h>
#include <iostream>
#include <stdint.h>
#include <limits.h>

#include "FSUPER.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------


void FSUPER::meanValue(const std::vector<FSUPER::pDescriptor> &descriptors,
  FSUPER::TDescriptor &mean)
{
	if(descriptors.empty()) return;

    if(descriptors.size() == 1)
    {
        mean = descriptors[0]->clone();
        return;
    }
//
    mean.create(1, descriptors[0]->cols,descriptors[0]->type());
	mean.setTo(cv::Scalar::all(0));
// 	float inv_s =1./double( descriptors.size());
// 	for(size_t i=0;i<descriptors.size();i++){
// 		mean +=  (*descriptors[i]) * inv_s;
// 	}

// 	mean.resize(0);
// 	mean.resize(FSUPER::L, 0);

	float s = descriptors.size();

// 	vector<FSUPER::pDescriptor>::const_iterator it;
// 	for(it = descriptors.begin(); it != descriptors.end(); ++it)
// 	{
// 		const FSUPER::TDescriptor &desc = **it;
//
// 		for(int i = 0; i < FSUPER::L; i += 4)
// 		{
// 			mean[i  ] += desc[i  ] / s;
// 			mean[i+1] += desc[i+1] / s;
// 			mean[i+2] += desc[i+2] / s;
// 			mean[i+3] += desc[i+3] / s;
// 		}
//
// 	}


    for(size_t i = 0; i < descriptors.size(); ++i)
    {
		const cv::Mat &d = *descriptors[i];
		const float *p = d.ptr<float>();
		for(int j = 0; j < d.cols; ++j, ++p){
// 			float aux = *p / s;
			mean.at<float>(0,j) += *p / s;

		}

    }

}

// --------------------------------------------------------------------------


double FSUPER::distance(const FSUPER::TDescriptor &a, const FSUPER::TDescriptor &b)
{
	double sqd = 0.;
	const float *a_ptr=a.ptr<float>(0);
	const float *b_ptr=b.ptr<float>(0);
	for(int i = 0; i < a.cols; i ++){
		sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
	}
	return sqd;
}

// --------------------------------------------------------------------------

std::string FSUPER::toString(const FSUPER::TDescriptor &a)
{

	stringstream ss;
	ss <<a.type()<<" "<<a.cols<<" ";

	const float *p = a.ptr<float>();
	for(int i = 0; i < a.cols; ++i, ++p)
		ss <<  *p << " ";

	return ss.str();
}

// --------------------------------------------------------------------------

void FSUPER::fromString(FSUPER::TDescriptor &a, const std::string &s)
{

	int type,cols;
	stringstream ss(s);
	ss >>type>>cols;

	a.create(1,  cols, type);
	float *p = a.ptr<float>();
	float n;
	for(int i = 0; i <  a.cols; ++i, ++p)
		if ( ss >> n) *p = (float)n;

}

// --------------------------------------------------------------------------

void FSUPER::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
	if(descriptors.empty())
	{
		mat.release();
		return;
	}

	const int N = descriptors.size();
	const int L = FSUPER::L;

	mat.create(N, L, CV_32F);

	for(int i = 0; i < N; ++i)
		memcpy(mat.ptr<float>(i),descriptors[i].ptr<float>(0),sizeof(float)*L);
}

// --------------------------------------------------------------------------

// void FORB::toMat8U(const std::vector<TDescriptor> &descriptors,
//   cv::Mat &mat)
// {
//   mat.create(descriptors.size(), 32, CV_8U);
//
//   unsigned char *p = mat.ptr<unsigned char>();
//
//   for(size_t i = 0; i < descriptors.size(); ++i, p += 32)
//   {
//     const unsigned char *d = descriptors[i].ptr<unsigned char>();
//     std::copy(d, d+32, p);
//   }
//
// }

// --------------------------------------------------------------------------

} // namespace DBoW2
