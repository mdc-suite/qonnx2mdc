
#pragma once
#include <hls_stream.h>
#include <cassert>

//Considerare di creare un array array[heads][tokens][tokens] per generare la matrixe A=QxKt
//e poi fare la partition completa della dimensione heads
//#pragma HLS ARRAY_PARTITION variable=array complete dim=0





/*
	 *  ____________		 ___________
	 * |			|		|			|
	 * |			|		|			|
	 A |			|		|			|
     * |			|  X  B |			|
	 * |			|		|			|
	 * |____________|       |___________|
 * 		|	|	|	|			  C
 *
 *
 *
 *
 *
 *
 */






//------------------------------------//
template<
	int A,
    int B,
    int C,
    int D,
    typename ActType,
    typename CoeffType
>
void LinearTokenBatchReshape(
    hls::stream<ActType> &input,
    hls::stream<ActType> (&output)[D],
    const CoeffType W[B][C],
    const CoeffType Bias[C]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bias complete dim=1

    static_assert(C % D == 0, "Embedded dimension E not divisible by H");
    const int P = C / D;

    ActType acc[A][C];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=output complete dim=1


    init:for (int t = 0; t < A; t++) {
        for (int o = 0; o < C; o++) {
            acc[t][o] = Bias[o];
        }
    }

    body:for (int t = 0; t < A; t++) {
        for (int e = 0; e < B; e++) {
            ActType val = input.read();
            //printf("Linear input value: %f\n", (float)val);
            for (int o = 0; o < C; o++) {
                acc[t][o] += W[e][o] * val;
            }
        }
    }

    write:for (int t = 0; t < A; t++) {
    	//printf("[");

        for (int p = 0; p < P; p++) {

            for (int h = 0; h < D; h++) {
				#pragma HLS UNROLL
                output[h].write(acc[t][h * P + p]);
                //if(h==0)
                //printf("%f,", (float)(acc[t][h * P + p]));
            }
        }
    	//printf("]\n");

    }
}


//-----------------------------------------------------//
template<
	int A,
	int B,
	int C,
	int D,
	typename ActType,
	typename CoeffType
>
void LinearTokenBatchReshapeTranspose(
    hls::stream<ActType> &input,
    hls::stream<ActType> (&output)[D],
    const CoeffType W[B][C],
    const CoeffType Bias[C]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bias complete dim=1

    static_assert(C % D == 0, "Embedded dimension E not divisible by H");
    const int P = C / D;

    ActType acc[A][C];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

    init: for (int t = 0; t < A; t++) {
        for (int o = 0; o < C; o++) {
            acc[t][o] = Bias[o];
        }
    }

    body:for (int t = 0; t < A; t++) {
            for (int e = 0; e < B; e++) {
                ActType val = input.read();
                //printf("Linear input value: %f\n", (float)val);
                for (int o = 0; o < C; o++) {
                    acc[t][o] += W[e][o] * val;
                }
            }
        }

    write: for (int p = 0; p < P; p++) {
    	//printf("[");
        for (int t = 0; t < A; t++) {
            for (int h = 0; h < D; h++) {
#pragma HLS UNROLL
                output[h].write(acc[t][h * P + p]);
                //if(h==0)
                //printf("%f ,",(float)acc[t][h * P + p]);


            }
        }
        //printf("]\n");
    }

}
//-----------------------------------------------------//

template<
	int A,
    int B,
    int C,
    int D,
    typename ActType,
    typename CoeffType
>
void MatMul(
    hls::stream<ActType> &matrix1,
    hls::stream<ActType> &matrix2,
    hls::stream<ActType> &output
) {
#pragma HLS INLINE off
	static_assert(B == D, "Input matrices dimension mismatch");  // <- FIXED with semicolon

    ActType matrix_A[A][B];
    ActType matrix_Out[A][C];
#pragma HLS ARRAY_PARTITION variable=matrix_A complete dim=1
#pragma HLS ARRAY_PARTITION variable=matrix_Out complete dim=1

    for (int a = 0; a < A; a++) {
        for (int b = 0; b < B; b++) {
            matrix_A[a][b] = matrix1.read();
        }
    }

    for (int a = 0; a < A; a++) {
        for (int c = 0; c < C; c++) {
            matrix_Out[a][c] = 0;
        }
    }

    for (int c = 0; c < C; c++) {
        for (int b = 0; b < B; b++) {
            ActType val = matrix2.read();
            for (int a = 0; a < A; a++) {
                matrix_Out[a][c] += matrix_A[a][b] * val;
            }
        }
    }

    for (int a = 0; a < A; a++) {
#pragma HLS PIPELINE II=1
        for (int c = 0; c < C; c++) {
            output.write(matrix_Out[a][c]);
        }
    }
}

//-----------------------------------------------------//

template<
    int M,                    // Output size
    int N,                    // Input size
    typename ActType,         // Activation (input/output) type
    typename CoeffType        // Weight/Bias type
>
void LinearBasic(
    hls::stream<ActType> &input,
    hls::stream<ActType> &output,
    const CoeffType W[M][N],  // Weights
    const CoeffType B[M]      // Biases
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1

    ActType acc[M];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    init: for (int m = 0; m < M; m++) {
#pragma HLS UNROLL
        acc[m] = 0;
    }

    body: for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
        ActType val = input.read();
        for (int m = 0; m < M; m++) {
#pragma HLS UNROLL
            acc[m] += W[m][n] * val;
        }
    }

    write: for (int m = 0; m < M; m++) {
        output.write(acc[m] + B[m]);
    }
}

//-----------------------------------------------------//
template<
	int TI,
	int A,
	int B,
	int C,
	typename ActType,
	typename CoeffType
>
void LinearTokenBatch(
    hls::stream<ActType> &input,
    hls::stream<ActType> &output,
    const CoeffType W[B][C],
    const CoeffType Bias[C]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bias complete dim=1


    ActType acc[A][C];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1


    init: for (int t = 0; t < A; t++) {
        for (int o = 0; o < C; o++) {
            acc[t][o] = Bias[o];
        }
    }

    body: for (int t = 0; t < A; t++) {
        for (int e = 0; e < B; e++) {
            ActType val = input.read();
            for (int o = 0; o < C; o++) {
                acc[t][o] += W[e][o] * val;
            }
        }
    }

    write: for (int p = 0; p < A; p++) {
        for (int t = 0; t < C; t++) {
                output.write(acc[p][t]);
            }
        }

}


//------------------------------------//
//
//
// These are just ideas, will need to work on them
/////////// NOT GOOD ///////////////
template<
	int I_h,
	int I_w,
	int O_w,
	int T_h,
	int T_w,
	int SIMD,
	int PE,
	int Heads,
    typename ActType,
    typename CoeffType,
	typename AccType
>
void LinearTiledStreaming(
    hls::stream<ActType> &input,
    hls::stream<AccType> &output,
    const CoeffType W[I_w][O_w],
    const CoeffType Bias[O_w]
) {

	//Assuming that the number of heads H is that I_h / H = M
	//we can deal with a mismatch with zero padding

	static_assert((I_h % Heads) == 0, "I_h must be divisible by Heads");
	static_assert((O_w % Heads) == 0, "O_w must be divisible by Heads");
	static_assert((O_w / Heads) == (I_h / Heads), "Per-head O_w must equal per-head I_h (M)");
	static_assert((T_h % (I_h/Heads)) == 0 || T_h == (I_h/Heads),
	              "Choose T_h so tiles align to M to avoid partial heads");

	ActType val;
	AccType acc[T_h][O_w];


	#pragma HLS ARRAY_PARTITION variable=acc dim=2 type=cyclic factor=PE

	#pragma HLS ARRAY_PARTITION variable=W    dim=2 type=cyclic factor=PE
	#pragma HLS ARRAY_PARTITION variable=Bias dim=1 type=cyclic factor=PE


	body: for(int i_h = 0; i_h < I_h; i_h+=T_h){

		init: for(int ii_h = 0; ii_h < T_h; ii_h++){
				for(int oo_w = 0; oo_w < O_w; oo_w++){
					#pragma HLS UNROLL factor=PE
					acc[ii_h][oo_w] = (AccType)Bias[oo_w];
					}
				}


		compute: for(int ii_h = 0; ii_h < T_h; ii_h++){
			for(int k=0; k < I_w; k++){
				input.read(val);
				for(int oo_w = 0; oo_w < O_w; oo_w++){
					#pragma HLS UNROLL factor=PE
					acc[ii_h][oo_w] += val*W[k][oo_w];
				}
			}
		}

		write: for(int ii_h = 0; ii_h < T_h; ii_h++){
			for(int oo_w = 0; oo_w < O_w; oo_w++){
				#pragma HLS PIPELINE II=1
				output.write(acc[ii_h][oo_w]);
				}
			}
		}


	}
//-----------------------------------------------------//
template<
	int T_h,
	int I_w,
	int SIMD,
	int PE,
	int Heads,
	int T_w,
    typename ActType
	>
void LinearSplitterStreaming(
    hls::stream<ActType> &input,
    hls::stream<ActType> (&output)[Heads]
) {

	  static_assert(I_w % Heads == 0, "I_w must be divisible by Heads");
	  static_assert(I_w == Heads * T_w, "I_w must equal Heads * T_w");
	  static_assert(T_h > 0 && T_w > 0, "T_h and T_w must be positive");
	  // For perfect MxM tiles to softmax with no buffering:
	  static_assert(T_h == T_w, "Set T_h == T_w (M) for square MxM tiles");

	  // Stream order: row-major; for each row, emit H contiguous chunks
	  RowLoop:
	  for (int r = 0; r < T_h; ++r) {
	    HeadLoop:
	    for (int h = 0; h < Heads; ++h) {
	      ChunkLoop:
	      for (int c = 0; c < T_w; ++c) {
	#pragma HLS PIPELINE II=1
	        ActType val = input.read();
	        output[h].write(val);
	      }
	    }
	  }
	}
//---------------------------------------------------

//The MLP is the absolute bottleneck, and I would like to
//highly vectorize and parallelize it. Let's try it, maybe implementing tiled
//operations?
// I can use ap_uint<...> a = w.range(High, Low); to pack and unpack data
//-----------------------------------------------------//
template<
	int A,
	int B,
	int C,
	int PF,
	typename ActType,
	typename CoeffType
>
void LinearTokenBatchVectorized(
    hls::stream<ap_uint<ActType::width * PF>> &input,
    hls::stream<ap_uint<ActType::width * PF>> &output,
    const CoeffType W[B][C],
    const CoeffType Bias[C]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bias complete dim=1


    ActType acc[A][C];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    init: for (int t = 0; t < A; t++) {
        for (int o = 0; o < C; o++) {
            acc[t][o] = Bias[o];
        }
    }

    body: for (int t = 0; t < A; t++) {
        for (int e = 0; e < B; e++) {
            ActType val = input.read();
            for (int o = 0; o < C; o++) {
                acc[t][o] += W[e][o] * val;
            }
        }
    }

    write: for (int p = 0; p < A; p++) {
        for (int t = 0; t < C; t++) {
                output.write(acc[p][t]);
            }
        }

}


//------------------------------------//

//TO DO!!!!
/*
 * #ifndef HLS_DEBUG
#define HLS_DEBUG 0
#endif

#if HLS_DEBUG
  #include <cstdio>
  #define DBG_PRINTF(...) std::printf(__VA_ARGS__)
#else
  #define DBG_PRINTF(...) do {} while (0)
#endif
 *
 *
 *
 *
 *
 * for (int t = 0; t < A; t++) {
    DBG_PRINTF("[");
    for (int p = 0; p < P; p++) {
        for (int h = 0; h < D; h++) {
#pragma HLS UNROLL
            output[h].write(acc[t][h * P + p]);
            if (h == 0) DBG_PRINTF("%f,", (float)acc[t][h * P + p]);
        }
    }
    DBG_PRINTF("]\n");
}
 * */


