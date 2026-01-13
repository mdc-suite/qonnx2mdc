#pragma once
#include <hls_stream.h>
#include <cassert>
#include <limits>
#include <ap_fixed.h>   // must be present




//-------------------------------------------------------------//
template<typename AccType, int LUT_SIZE>
AccType exp_lut_lookup(AccType z) {
#pragma HLS INLINE

    // --- Configurable range ---
    const AccType Z_MIN = (AccType)(-8.0);
    const AccType Z_MAX = (AccType)0.0;
    const AccType STEP  = (Z_MAX - Z_MIN) / (AccType)(LUT_SIZE - 1);
    const AccType INV_STEP = (AccType)1.0 / STEP;

    // --- Precomputed exp(z) values over [-8,0] ---
    // TODO: generate these offline (Python/Matlab) and paste.

    // exp(z) LUT, z in [-8.0, 0.0], 256 entries
static const AccType LUT[256] = {
    (AccType)3.3546262790e-04, (AccType)3.4615377301e-04, (AccType)3.5718564335e-04, (AccType)3.6856909779e-04, 
    (AccType)3.8031534127e-04, (AccType)3.9243593582e-04, (AccType)4.0494281195e-04, (AccType)4.1784828041e-04, 
    (AccType)4.3116504426e-04, (AccType)4.4490621144e-04, (AccType)4.5908530761e-04, (AccType)4.7371628952e-04, 
    (AccType)4.8881355869e-04, (AccType)5.0439197563e-04, (AccType)5.2046687445e-04, (AccType)5.3705407795e-04, 
    (AccType)5.5416991322e-04, (AccType)5.7183122766e-04, (AccType)5.9005540562e-04, (AccType)6.0886038548e-04, 
    (AccType)6.2826467731e-04, (AccType)6.4828738109e-04, (AccType)6.6894820553e-04, (AccType)6.9026748743e-04, 
    (AccType)7.1226621175e-04, (AccType)7.3496603220e-04, (AccType)7.5838929263e-04, (AccType)7.8255904896e-04, 
    (AccType)8.0749909190e-04, (AccType)8.3323397038e-04, (AccType)8.5978901569e-04, (AccType)8.8719036642e-04, 
    (AccType)9.1546499422e-04, (AccType)9.4464073028e-04, (AccType)9.7474629281e-04, (AccType)1.0058113152e-03, 
    (AccType)1.0378663754e-03, (AccType)1.0709430256e-03, (AccType)1.1050738239e-03, (AccType)1.1402923658e-03, 
    (AccType)1.1766333175e-03, (AccType)1.2141324499e-03, (AccType)1.2528266743e-03, (AccType)1.2927540780e-03, 
    (AccType)1.3339539623e-03, (AccType)1.3764668808e-03, (AccType)1.4203346799e-03, (AccType)1.4656005393e-03, 
    (AccType)1.5123090151e-03, (AccType)1.5605060832e-03, (AccType)1.6102391848e-03, (AccType)1.6615572732e-03, 
    (AccType)1.7145108615e-03, (AccType)1.7691520730e-03, (AccType)1.8255346920e-03, (AccType)1.8837142168e-03, 
    (AccType)1.9437479146e-03, (AccType)2.0056948776e-03, (AccType)2.0696160813e-03, (AccType)2.1355744445e-03, 
    (AccType)2.2036348911e-03, (AccType)2.2738644142e-03, (AccType)2.3463321420e-03, (AccType)2.4211094057e-03, 
    (AccType)2.4982698099e-03, (AccType)2.5778893050e-03, (AccType)2.6600462617e-03, (AccType)2.7448215488e-03, 
    (AccType)2.8322986119e-03, (AccType)2.9225635562e-03, (AccType)3.0157052312e-03, (AccType)3.1118153178e-03, 
    (AccType)3.2109884189e-03, (AccType)3.3133221522e-03, (AccType)3.4189172468e-03, (AccType)3.5278776416e-03, 
    (AccType)3.6403105883e-03, (AccType)3.7563267567e-03, (AccType)3.8760403435e-03, (AccType)3.9995691848e-03, 
    (AccType)4.1270348724e-03, (AccType)4.2585628728e-03, (AccType)4.3942826515e-03, (AccType)4.5343277998e-03, 
    (AccType)4.6788361665e-03, (AccType)4.8279499938e-03, (AccType)4.9818160571e-03, (AccType)5.1405858095e-03, 
    (AccType)5.3044155307e-03, (AccType)5.4734664813e-03, (AccType)5.6479050611e-03, (AccType)5.8279029730e-03, 
    (AccType)6.0136373921e-03, (AccType)6.2052911402e-03, (AccType)6.4030528652e-03, (AccType)6.6071172277e-03, 
    (AccType)6.8176850917e-03, (AccType)7.0349637228e-03, (AccType)7.2591669923e-03, (AccType)7.4905155874e-03, 
    (AccType)7.7292372284e-03, (AccType)7.9755668934e-03, (AccType)8.2297470490e-03, (AccType)8.4920278891e-03, 
    (AccType)8.7626675814e-03, (AccType)9.0419325213e-03, (AccType)9.3300975942e-03, (AccType)9.6274464460e-03, 
    (AccType)9.9342717623e-03, (AccType)1.0250875557e-02, (AccType)1.0577569468e-02, (AccType)1.0914675067e-02, 
    (AccType)1.1262524172e-02, (AccType)1.1621459178e-02, (AccType)1.1991833389e-02, (AccType)1.2374011373e-02, 
    (AccType)1.2768369314e-02, (AccType)1.3175295385e-02, (AccType)1.3595190130e-02, (AccType)1.4028466860e-02, 
    (AccType)1.4475552056e-02, (AccType)1.4936885793e-02, (AccType)1.5412922168e-02, (AccType)1.5904129753e-02, 
    (AccType)1.6410992052e-02, (AccType)1.6934007980e-02, (AccType)1.7473692348e-02, (AccType)1.8030576379e-02, 
    (AccType)1.8605208223e-02, (AccType)1.9198153500e-02, (AccType)1.9809995856e-02, (AccType)2.0441337540e-02, 
    (AccType)2.1092799990e-02, (AccType)2.1765024455e-02, (AccType)2.2458672615e-02, (AccType)2.3174427241e-02, 
    (AccType)2.3912992863e-02, (AccType)2.4675096463e-02, (AccType)2.5461488195e-02, (AccType)2.6272942116e-02, 
    (AccType)2.7110256956e-02, (AccType)2.7974256897e-02, (AccType)2.8865792391e-02, (AccType)2.9785740992e-02, 
    (AccType)3.0735008220e-02, (AccType)3.1714528457e-02, (AccType)3.2725265861e-02, (AccType)3.3768215318e-02, 
    (AccType)3.4844403424e-02, (AccType)3.5954889487e-02, (AccType)3.7100766580e-02, (AccType)3.8283162608e-02, 
    (AccType)3.9503241426e-02, (AccType)4.0762203978e-02, (AccType)4.2061289484e-02, (AccType)4.3401776655e-02, 
    (AccType)4.4784984957e-02, (AccType)4.6212275907e-02, (AccType)4.7685054411e-02, (AccType)4.9204770152e-02, 
    (AccType)5.0772919012e-02, (AccType)5.2391044548e-02, (AccType)5.4060739509e-02, (AccType)5.5783647405e-02, 
    (AccType)5.7561464125e-02, (AccType)5.9395939605e-02, (AccType)6.1288879552e-02, (AccType)6.3242147219e-02, 
    (AccType)6.5257665242e-02, (AccType)6.7337417531e-02, (AccType)6.9483451223e-02, (AccType)7.1697878696e-02, 
    (AccType)7.3982879650e-02, (AccType)7.6340703252e-02, (AccType)7.8773670348e-02, (AccType)8.1284175752e-02, 
    (AccType)8.3874690598e-02, (AccType)8.6547764774e-02, (AccType)8.9306029435e-02, (AccType)9.2152199589e-02, 
    (AccType)9.5089076772e-02, (AccType)9.8119551803e-02, (AccType)1.0124660763e-01, (AccType)1.0447332227e-01, 
    (AccType)1.0780287184e-01, (AccType)1.1123853367e-01, (AccType)1.1478368954e-01, (AccType)1.1844182901e-01, 
    (AccType)1.2221655286e-01, (AccType)1.2611157660e-01, (AccType)1.3013073418e-01, (AccType)1.3427798174e-01, 
    (AccType)1.3855740146e-01, (AccType)1.4297320567e-01, (AccType)1.4752974092e-01, (AccType)1.5223149228e-01, 
    (AccType)1.5708308777e-01, (AccType)1.6208930290e-01, (AccType)1.6725506539e-01, (AccType)1.7258545997e-01, 
    (AccType)1.7808573344e-01, (AccType)1.8376129984e-01, (AccType)1.8961774571e-01, (AccType)1.9566083565e-01, 
    (AccType)2.0189651799e-01, (AccType)2.0833093062e-01, (AccType)2.1497040705e-01, (AccType)2.2182148262e-01, 
    (AccType)2.2889090097e-01, (AccType)2.3618562066e-01, (AccType)2.4371282200e-01, (AccType)2.5147991415e-01, 
    (AccType)2.5949454239e-01, (AccType)2.6776459567e-01, (AccType)2.7629821435e-01, (AccType)2.8510379821e-01, 
    (AccType)2.9419001474e-01, (AccType)3.0356580767e-01, (AccType)3.1324040576e-01, (AccType)3.2322333188e-01, 
    (AccType)3.3352441241e-01, (AccType)3.4415378687e-01, (AccType)3.5512191794e-01, (AccType)3.6643960176e-01, 
    (AccType)3.7811797851e-01, (AccType)3.9016854342e-01, (AccType)4.0260315809e-01, (AccType)4.1543406212e-01, 
    (AccType)4.2867388518e-01, (AccType)4.4233565949e-01, (AccType)4.5643283254e-01, (AccType)4.7097928046e-01, 
    (AccType)4.8598932155e-01, (AccType)5.0147773046e-01, (AccType)5.1745975271e-01, (AccType)5.3395111968e-01, 
    (AccType)5.5096806412e-01, (AccType)5.6852733609e-01, (AccType)5.8664621951e-01, (AccType)6.0534254910e-01, 
    (AccType)6.2463472800e-01, (AccType)6.4454174583e-01, (AccType)6.6508319742e-01, (AccType)6.8627930208e-01, 
    (AccType)7.0815092350e-01, (AccType)7.3071959032e-01, (AccType)7.5400751726e-01, (AccType)7.7803762705e-01, 
    (AccType)8.0283357293e-01, (AccType)8.2841976200e-01, (AccType)8.5482137919e-01, (AccType)8.8206441207e-01, 
    (AccType)9.1017567645e-01, (AccType)9.3918284273e-01, (AccType)9.6911446317e-01, (AccType)1.0000000000e+00, 
};

    // --- Clamp z into [Z_MIN, Z_MAX] ---
    if (z < Z_MIN) z = Z_MIN;
    if (z > Z_MAX) z = Z_MAX;

    // Map z to [0, LUT_SIZE-1]:  idx ≈ (z - Z_MIN) / STEP
    AccType t = (z - Z_MIN) * INV_STEP;
    int idx = (int)t;
    if (idx < 0) idx = 0;
    if (idx > LUT_SIZE - 1) idx = LUT_SIZE - 1;

    return LUT[idx];
}
//-------------------------------------------------------------//
//-------------------------------------------------------------//
template<
    int A,  // rows of Q
    int L,  // cols of Q, rows of K
    int J,  // cols of K
    typename ActType
>
void Softmax_LUT(
    hls::stream<ActType> &Q,
    hls::stream<ActType> &K,
    hls::stream<ActType> &V,
    hls::stream<ActType> &output
){
#pragma HLS INLINE off

    typedef ap_fixed<32,16,AP_RND,AP_SAT> AccType;

    // You can tune this
    const int LUT_SIZE = 256;

    // Matrices / buffers
    AccType  A_m[A][J];      // logits (QK^T)
    AccType  Out_m[A][L];    // output
    ActType  Q_m[A][L];
    ActType  K_m[L][J];
    ActType  V_m[A][L];

    ActType  MAX_BUF[A];     // row-wise max of logits
    AccType  row_sum[A];     // Σ_j exp(logit_j - max)
    ActType  soft_w[A][J];   // final softmax weights

#pragma HLS ARRAY_PARTITION variable=Out_m   complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_m     complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_m     complete dim=2
#pragma HLS ARRAY_PARTITION variable=K_m     complete dim=1
#pragma HLS ARRAY_PARTITION variable=MAX_BUF complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=soft_w  complete dim=1

#pragma HLS ARRAY_PARTITION variable=Out_m   complete dim=2

    // -------------------- Init --------------------
    init:
    for (int a = 0; a < A; a++) {
    #pragma HLS PIPELINE II=1
        MAX_BUF[a] = (ActType)(-1e9);  // very small
        row_sum[a] = (AccType)0;
        for (int l = 0; l < L; ++l)
            Out_m[a][l] = (AccType)0;
    }

    // -------------------- Load Q --------------------
    load_Q:
    for (int a = 0; a < A; a++) {
        for (int l = 0; l < L; l++) {
        #pragma HLS PIPELINE II=1
            Q_m[a][l] = Q.read();
        }
    }

    // -------------------- Load K --------------------
    load_K:
    for (int l = 0; l < L; l++) {
        for (int j = 0; j < J; j++) {
        #pragma HLS PIPELINE II=1
            K_m[l][j] = K.read();
        }
    }

    // -------------------- Load V --------------------
    load_V:
    for (int j = 0; j < J; j++) {
        for (int l = 0; l < L; l++) {
        #pragma HLS PIPELINE II=1
            V_m[j][l] = V.read();
        }
    }



    // -------------------- Q x K^T (logits) --------------------
    qk_matmul:
	//printf("MATRIX A\n");
    for (int a = 0; a < A; a++) {
    	//printf("[");
        for (int j = 0; j < J; j++) {
        #pragma HLS PIPELINE II=1
            AccType acc = 0;
            for (int l = 0; l < L; l++) {
            #pragma HLS UNROLL
                AccType q = (AccType)Q_m[a][l];
                AccType k = (AccType)K_m[l][j];
                acc += q * k;
                //printf("Q: %f -- K: %f \n", (float)q, (float)k);
            }
            A_m[a][j] = (AccType)(acc * Head_scale);

            //printf("%f ,", (float)A_m[a][j]);

            // Track row max for softmax stability
            if (A_m[a][j] > MAX_BUF[a]) {
                MAX_BUF[a] = A_m[a][j];
            }
        }
        //printf("]\n");
    }

    // -------------------- Softmax with LUT --------------------
    // For each row a:
    //   1) compute z_j = logit_j - max
    //   2) e_j = exp_lut(z_j)
    //   3) row_sum = Σ_j e_j
    //   4) soft_w_j = e_j / row_sum
    softmax_rows:
    for (int a = 0; a < A; ++a) {
        // Reset row_sum
        row_sum[a] = (AccType)0;

        // 1) exp(logit_j - max) via LUT
        compute_exp:
        for (int j = 0; j < J; ++j) {
        #pragma HLS PIPELINE II=1
            AccType x = (AccType)A_m[a][j];
            AccType m = (AccType)MAX_BUF[a];
            AccType z = x - m; // z ≤ 0

            AccType e = exp_lut_lookup<AccType, LUT_SIZE>(z);

            soft_w[a][j] = (ActType)e;   // store numerator for now
            row_sum[a] += e;
        }

        
        normalize_row:
        for (int j = 0; j < J; ++j) {
        #pragma HLS PIPELINE II=1
            AccType w = soft_w[a][j] / row_sum[a];
            soft_w[a][j] = (ActType)w;
        }
    }

    //printf("\n");
    //printf("SOFTMAX(QxKt/sqrt(dim))\n");
    for(int a = 0; a < A; a++){
    	//printf("[");
    	for(int j=0; j < J; j++){
    		//printf("%f ,", (float)soft_w[a][j]);
    	}
    	//printf("]\n");
    }



    // -------------------- A x V with softmax weights --------------------
    // Out[a,l] = Σ_j soft_w[a,j] * V[j,l]

    av_matmul:
    for (int a = 0; a < A; a++) {
        for (int l = 0; l < L; l++) {
            for (int j = 0; j < J; j++) {
                AccType w = (AccType) soft_w[a][j];
                AccType v = (AccType) V_m[j][l];
                Out_m[a][l] += w * v;
            }
        }
    }
  


    // -------------------- Write outputs --------------------
    write_out:
	//printf("MHA\n");
    for (int a = 0; a < A; a++) {
    	//printf("[");
        for (int l = 0; l < L; l++) {
        #pragma HLS PIPELINE II=1
            output.write(Out_m[a][l]);
            //printf("%f ,",(float)Out_m[a][l]);
        }
        //printf("]\n");
    }
}
//-----------------------------------------
//-------------------------------------------------------------//
template<
	int A,
    int L,
    typename ActType         // Activation (input/output) type
>
void RELU(
    hls::stream<ActType> &input,
    hls::stream<ActType> &output
){

	ActType val;

	for(int a=0; a<A;a++){
        //printf("[");
		for(int l=0; l<L;l++){
			val = input.read();
			output.write((val > 0) ? val : (ActType)0 );
            //printf("%f ,", (float)((val > 0) ? val : (ActType)0));
		}
        //printf("]\n");
	}


}

////// WORK IN PROGRESS: STREAMING ITA SOFTMAX ACCELERATORS IN VITIS HLS //////
///// NOT READY !!!! /////////////
//-------------------------------------------------------------//
template<
    int A,  // rows of Q
    int L,  // cols of Q, rows of K
    int J,  // cols of K
    typename ActType
>
void Softmax_ITA_fixed(
    hls::stream<ActType> &Q,
    hls::stream<ActType> &K,
    hls::stream<ActType> &V,
    hls::stream<ActType> &output
){
#pragma HLS INLINE off



    typedef ap_uint<15> acc_type;
    typedef ap_uint<16> inv_type;
    typedef ap_fixed<32,16,AP_RND,AP_SAT> AccType;


    static const ap_fixed<18,2,AP_RND,AP_SAT> log2e = 1.44269504089;


    ap_int<8>  A_m[A][J] = {0};
    AccType    Out_m[A][L] = {0};
    ActType    Q_m[A][L];
    ActType    K_m[L][J];
    ActType    V_m[A][L];

    ap_int<8>  MAX_BUF[A];
    acc_type  ACC[A];
    inv_type  Den[A];

    static const int SCALE = 256;

#pragma HLS ARRAY_PARTITION variable=Out_m complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_m   complete dim=1
#pragma HLS ARRAY_PARTITION variable=ACC   complete dim=1
#pragma HLS ARRAY_PARTITION variable=Den   complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_m   complete dim=2
#pragma HLS ARRAY_PARTITION variable=MAX_BUF complete dim=1

    // Init
    for (int a = 0; a < A; a++) {
#pragma HLS PIPELINE II=1
        MAX_BUF[a] = -128;
        ACC[a]     = (AccType)0;
    }

    // Load Q
    for (int a = 0; a < A; a++) {
        for (int l = 0; l < L; l++) {
#pragma HLS PIPELINE II=1
            Q_m[a][l] = Q.read();
            Out_m[a][l] = 0;
        }
    }

    // Load K
    for (int l = 0; l < L; l++) {
        for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
            K_m[l][j] = K.read();
        }
    }

    // Load V
    for (int a = 0; a < J; a++) {
            for (int l = 0; l < L; l++) {
    #pragma HLS PIPELINE II=1
                V_m[a][l] = V.read();
            }
        }



    // Q x K^T
    for (int a = 0; a < A; a++) {
		for (int j = 0; j < J; j++) {

			AccType acc = 0;
			for (int l = 0; l < L; l++) {
			#pragma HLS PIPELINE II=1
				acc += (AccType)(Q_m[a][l] * K_m[l][j]);
			}
			AccType t = acc * log2e;
			ap_int<16> q = (t >= 0) ? ap_int<16>(t + (ap_int<16>)0.5) : ap_int<16>(t - (ap_int<16>)0.5);

			// saturate to int8
			ap_int<8> xq;
			if (q > (ap_int<16>)(127)) A_m[a][j] = (ap_int<8>)(127);
			else if (q < (ap_int<16>)(-128)) A_m[a][j] = (ap_int<8>)(-128);
			else A_m[a][j] = (ap_int<8>)q;
		}
    }



    //DA
    for (int a = 0; a < A; a++) {
        for (int j = 0; j < J; j++) {
        	ap_int<8> xq = A_m[a][j];
            if (xq > MAX_BUF[a]) {
                ap_uint<9> delta = ((xq - MAX_BUF[a])); // Half up rounding
                if (delta >= 15)
                	ACC[a] = 0;
                else
                	ACC[a] >>= delta;

                ACC[a] += (acc_type)(SCALE);
                MAX_BUF[a] = xq;
            } else {
            	ap_uint<9> k = ((MAX_BUF[a] - xq));
            	// k >= 9 because if i shift by 9 bits, term = 0 anyway
            	ap_uint<16> term = (k >= (ap_uint<9>)9) ? (ap_uint<16>)(0): (((ap_uint<16>)SCALE >> k));
                ACC[a] += term;
            }
        }
    }
//-----------------------------STAGE 2-----------------------------//
    // DI: denominator inversion
    for (int a = 0; a < A; a++) {
#pragma HLS PIPELINE II=1
        if (ACC[a] == 0)
        	Den[a] = 0;
        else
        	Den[a] = (((ap_uint<16>)((SCALE - 1) * SCALE)) / ACC[a]);
    }

    // A x V + EN
    for (int j = 0; j < J; j++) {
        for (int l = 0; l < L; l++) {
#pragma HLS PIPELINE II=1
            ActType v = V_m[j][l];
            for (int a = 0; a < A; a++) {
            	ap_int<8> xq = A_m[a][j];
                // delta >= 0, k in [0..7]
            	//printf("Computing MAX_BUF[a] - xq: %f - %f\n", (float)(MAX_BUF[a]),(float)xq);
            	ap_uint<9> d = (ap_uint<9>)(MAX_BUF[a] - xq + (ap_int<8>)(0.5));
            	ap_uint<16> w_int = (d >= 15) ? (ap_uint<16>)(0) : (Den[a] >> d);  // 0..255 approx

            	typedef ap_fixed<32,16> prob_t;
            	prob_t w = prob_t(w_int) / prob_t(256);

            	//printf("PROB T %f\n",(float)w);

                ActType contrib = v * w;
                Out_m[a][l] += contrib;
                }
        }
    }

    // Write outputs
    for (int a = 0; a < A; a++) {
        for (int l = 0; l < L; l++) {
#pragma HLS PIPELINE II=1
            output.write(Out_m[a][l]);
        }
    }
}

