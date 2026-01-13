#include <hls_stream.h>
#include "ViT_Block.hpp"
#include <hls_math.h>
#include <limits.h>
#include <cmath>
#include <cfloat>   // for FLT_MAX
#include <fstream>
#include <iostream>

#define STREAM_FROM_FILE 1

#define INPUT_SIZE 28*28 //MNIST default
#define OUTPUT_SIZE 10 //MNIST default
void initialize_stream_file(const char* file_name, stream<act_0> &in_stream);
void SOFTMAX(hls::stream<act_0> &in, hls::stream<float> &out, int N);

int main(){

	hls::stream<act_0> input("inputvit");
	hls::stream<act_0> output("outputvit");
	hls::stream<act_0> softmax_i("inp_soft");
	hls::stream<float> softmax_o("softmax_outputvit");
	act_0 val[OUTPUT_SIZE];

	#if STREAM_FROM_FILE
    const char *s = "/home/fede/PhD/ViT_Brevitas/notebook/patch_output.txt";
    printf("Initializing stream with file: %s \n", s);
    initialize_stream_file(s, input);
    #else
	for(int i=0; i < INPUT_SIZE; i++){
		input.write((act_0)1);
	}
	#endif

	printf("Entering Vit Block\n");
	ViT_Block(input,output);

	printf("Output values:\n");
	int count = 0;
	while(!output.empty()){
		output.read(val[count]);
		softmax_i.write(val[count]);
		count++;
	}

	SOFTMAX(softmax_i, softmax_o, count);

	float val1;
	int cnt = 0;
	while(!softmax_o.empty()){
			softmax_o.read(val1);
			printf("Softmax in output: %f, actual value: %f\n", val1, (float)val[cnt]);
			cnt++;
		}


	return 0;
}


void initialize_stream_file(const char* file_name, stream<act_0> &in_stream){
            std::ifstream infile(file_name);  // Open the file
            if (!infile) {
                std::cerr << "Error: Cannot open file " << file_name << std::endl;
                return;
            }

            float value;
            while (infile >> value) {  // Read float value from the file
                act_0 converted_value = (act_0) value;  // Implicit conversion from float to ap_fixed
                in_stream.write(converted_value);  // Send to the stream
            }

            infile.close();  // Close the file
        }

void SOFTMAX(hls::stream<act_0> &in, hls::stream<float> &out, int N) {

    // Basic floating-point Softmax function, just used to compute output values
    // softmax(x_i) = e^(x_i) / sum_j e^(x_j), j = 0 ... N-1

    float array_inputs[N];   // OK for sim (VLA), not for synthesis
    float exp_vals[N];       // store exponentials so we don't recompute

    float accumulator = 0.0f;
    float max_val;
    act_0 temp;

    // Initialize max to a very small float
    max_val = -FLT_MAX;

    // 1) Read inputs, find max, and store inputs as float
    for (int i = 0; i < N; i++) {
        in.read(temp);
        float x = (float)temp;
        array_inputs[i] = x;
        max_val = (x > max_val) ? x : max_val;
    }

    // 2) Compute exponentials of (x - max) and accumulate
    for (int i = 0; i < N; i++) {
        float e = std::exp(array_inputs[i] - max_val);
        exp_vals[i] = e;
        accumulator += e;
    }

    // 3) Normalize and write softmax outputs
    for (int i = 0; i < N; i++) {
        out.write(exp_vals[i] / accumulator);
    }
}
