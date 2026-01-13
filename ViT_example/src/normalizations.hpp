#pragma once
#include <hls_stream.h>
#include <cassert>
#include <limits>
#include "types.h"
#include "activations.h"
#include "linear.h"
#include "ap_fixed.h"


//---------------------------------------------------------//


template<
	int A, //Tokens
	int B, //Embedding dimension
	typename ActType,
	typename CoeffType
>
void ADDNORM(
		hls::stream<ActType> &residual,
		hls::stream<ActType> &input,
		hls::stream<ActType> &output,
		const CoeffType running_mean[B],
		const CoeffType running_std[B],
		const CoeffType weight[B],
		const CoeffType bias[B]
		){
	//In this block we need to add the residual to the actual input, and then normalize it.
	//It could make sense to add some kind of template to select which normalization function we want to use, but
	//for now I will stick to BatchNorm.

	//The MHA writes T tokens x E embeddimg_dimension = T x I dimension for each head x H number of heads
	//The MLP writes the same. So, we need in input the T and E parameters


	ActType res, val, mean,std,scale,Bias;
	ap_int<16> sum[A][B];
	ActType out;

	for(int a = 0; a < A; a++){
		for(int b = 0; b < B; b++){
			res = residual.read();
			val = input.read();
			mean = running_mean[b];
			std = running_std[b];
			scale = weight[b];
			Bias = bias[b];

			out = (((res+val) - mean) * std) * scale + Bias;
			output.write((ActType)out);
		}
	}




}

//---------------------------------------------------------//


template<
	int A, //Tokens
	int B, //Embedding dimension
	typename InType,
	typename OutType,
	typename CoeffType
>
void BATCHNORM(
		hls::stream<InType> &input,
		hls::stream<OutType> &output,
		const CoeffType running_mean[B],
		const CoeffType running_std[B],
		const CoeffType weight[B],
		const CoeffType bias[B]
		){
	//In this block we need to perform Batch Normalization
	//Parameters are already saved on-chip


	InType val;
	CoeffType mean,std,scale,Bias;
	OutType out;

	for(int a = 0; a < A; a++){
		//printf("[");
		for(int b = 0; b < B; b++){
			val = input.read();
			//printf("BatchNorm input: %f\n", (float)val);
			mean = running_mean[b];
			std = running_std[b];
			scale = weight[b];
			Bias = bias[b];
			//printf("Mean: %f , std: %f , scale: %f , Bias:%f \n", (float)mean,(float)std,(float)scale,(float)Bias);

			out = (OutType)(((val - mean) * std) * scale + Bias);
			//printf("%f ,",(float)out);
			//printf("BatchNorm out: %f \n",(float)out);
			output.write((OutType)out);
		}

		//printf("]\n");
	}
}

//---------------------------------------------------------//


template<
	int A, //Tokens
	int B, //Embedding dimension
	typename ActType
	>
void RESIDUAL(
		hls::stream<ActType> &residual,
		hls::stream<ActType> &input,
		hls::stream<ActType> &output
		){
	//In this block we need to add the residual to the actual input.


	ActType res, val;
	ActType out;

	
	for(int a = 0; a < A; a++){
		#pragma HLS LOOP_FLATTEN
		//printf("riga %d [", a);
		for(int b = 0; b < B; b++){
			res = residual.read();
			val = input.read();

			

			out = res+val;
			//printf("%f ,",(float)out);
			output.write((ActType)out);
		}

		//printf("]\n");
	}




}
