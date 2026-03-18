#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <string.h>

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t _err = (call);                                 \
        if (_err != cudaSuccess)                                   \
        {                                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

__global__ void Feedforward(int batch_size, int in_features, int out_features,
                            float *X, float *W, float *B, float *Out)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < batch_size && col < out_features)
    {
        float dot = 0.f;
        for (int i = 0; i < in_features; i++)
            dot += X[row * in_features + i] * W[i * out_features + col];
        Out[row * out_features + col] = dot + B[col];
    }
}

__global__ void FeedforwardReLU(int batch_size, int in_features, int out_features,
                            float *X, float *W, float *B, float *Z)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < batch_size && col < out_features)
    {
        float dot = B[col];
        for (int i = 0; i < in_features; i++)
            dot += X[row * in_features + i] * W[i * out_features + col];
        Z[row * out_features + col] = dot > 0 ? dot : 0 ;
    }

}

__global__ void Softmax(int R, int C, float *input, float *output)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < R)
    {
        float max_val = input[row * C];
        for (int i = 1; i < C; i++)
            max_val = fmaxf(max_val, input[row * C + i]);
        float denom = 0.f;
        for (int i = 0; i < C; i++)
            denom += expf(input[row * C + i] - max_val);
        for (int i = 0; i < C; i++)
            output[row * C + i] = expf(input[row * C + i] - max_val) / denom;
    }
}


__global__ void crossEntropyLoss(int R, int C, float *pred, float *real, float *output)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < R)
    {
        float loss = 0.f;
        for (int i = 0; i < C; i++)
            loss -= real[row * C + i] * logf(fmaxf(1e-7f, pred[row * C + i]));
        output[row] = loss;
    }
}

__global__ void init_weights(int R, int C, float *M, unsigned long long seed)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < R && col < C)
    {
        curandState state;
        curand_init(seed, row * C + col, 0, &state);
        M[row * C + col] = curand_normal(&state) * sqrtf(2.f / R);
    }
}

__global__ void crossEntropyBackwards(int R, int C, float *pred, float *real, float *output)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < R && col < C)
        output[row * C + col] = pred[row * C + col] - real[row * C + col];
}

__global__ void feedforwarrelu_backward(
    int batch_size, int in_features, int out_features,
    float *X, float *Z, float *dL, float *dW, float *dB)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < in_features && col < out_features)
    {
        float dw_sum = 0.f, db_sum = 0.f;
        for (int b = 0; b < batch_size; b++)
        {
            float dz = (Z[b * out_features + col] > 0.f) ? dL[b * out_features + col] : 0.f;
            dw_sum += X[b * in_features + row] * dz;
            if (row == 0)
                db_sum += dz;
        }
        dW[row * out_features + col] = dw_sum;
        if (row == 0)
            dB[col] = db_sum;
    }
}

__global__ void feedforwarlinear_backward(
    int batch_size, int in_features, int out_features,
    float *X, float *dZ, float *dW, float *dB)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < in_features && col < out_features)
    {
        float dw_sum = 0.f, db_sum = 0.f;
        for (int b = 0; b < batch_size; b++)
        {
            dw_sum += X[b * in_features + row] * dZ[b * out_features + col];
            if (row == 0)
                db_sum += dZ[b * out_features + col];
        }
        dW[row * out_features + col] = dw_sum;
        if (row == 0)
            dB[col] = db_sum;
    }
}

__global__ void feedforwarrelu_backwardx(
    int batch_size, int in_features, int out_features,
    float *Z, float *W, float *dL, float *dX)
{
    int b = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    if (b < batch_size && i < in_features)
    {
        float sum = 0.f;
        for (int j = 0; j < out_features; j++)
        {
            float dz = (Z[b * out_features + j] > 0.f) ? dL[b * out_features + j] : 0.f;
            sum += dz * W[i * out_features + j];
        }
        dX[b * in_features + i] = sum;
    }
}

__global__ void feedforwarlinear_backwardx(
    int batch_size, int in_features, int out_features,
    float *W, float *dZ, float *dX)
{
    int b = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    if (b < batch_size && i < in_features)
    {
        float sum = 0.f;
        for (int j = 0; j < out_features; j++)
            sum += dZ[b * out_features + j] * W[i * out_features + j];
        dX[b * in_features + i] = sum;
    }
}

__global__ void update_layer(int R, int C, int batch_size, float lr,
                             float *weights, float *biases, float *dW, float *dB)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row < R && col < C)
    {
        weights[row * C + col] -= lr * dW[row * C + col] / batch_size;
        if (row == 0)
            biases[col] -= lr * dB[col] / batch_size;
    }
}

bool loamnist_csv(const char *path,
                    float **images_out, float **labels_out, int *N_out,
                    bool skip_header = false)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return false;
    }

    int N = 0;
    char line[8192];
    if (skip_header)
        fgets(line, sizeof(line), f);
    while (fgets(line, sizeof(line), f))
        if (line[0] != '\n')
            N++;
    rewind(f);
    if (skip_header)
        fgets(line, sizeof(line), f);

    const int PX = 784;
    float *images = (float *)malloc(N * PX * sizeof(float));
    float *labels = (float *)calloc(N * 10, sizeof(float));
    if (!images || !labels)
    {
        fprintf(stderr, "malloc failed for %s\n", path);
        fclose(f);
        return false;
    }

    for (int i = 0; i < N; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;

        char *tok = strtok(line, ",");
        int label = atoi(tok);
        labels[i * 10 + label] = 1.f;

        for (int p = 0; p < PX; p++)
        {
            tok = strtok(nullptr, ",\n\r");
            images[i * PX + p] = tok ? atoi(tok) / 255.f : 0.f;
        }
    }

    fclose(f);
    *images_out = images;
    *labels_out = labels;
    *N_out = N;
    return true;
}

float mean_loss(float *loss, int N, float *h_buf)
{
    cudaMemcpy(h_buf, loss, N * sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.f;
    for (int i = 0; i < N; i++)
        s += h_buf[i];
    return s / N;
}

int argmax(const float *row, int C)
{
    int best = 0;
    for (int i = 1; i < C; i++)
        if (row[i] > row[best])
            best = i;
    return best;
}

void shuffle(int *arr, int N)
{
    for (int i = N - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

int main()
{

    const int BATCH = 128;
    const int EPOCHS = 10;
    const float LR = 0.01f;

    const int IN = 784;
    const int H1 = 256;
    const int H2 = 128;
    const int OUT = 10;

    float *h_train_X = nullptr, *h_train_Y = nullptr;
    float *h_test_X = nullptr, *h_test_Y = nullptr;
    int N_train = 0, N_test = 0;

    if (!loamnist_csv("data/mnist_train.csv",
                        &h_train_X, &h_train_Y, &N_train))
    {
        fprintf(stderr,
                "ERROR: Could not open data/mnist_train.csv\n"
                "Expected format: label,px0,px1,...,px783  (no header)\n");
        return 1;
    }

    bool have_test = loamnist_csv("data/mnist_test.csv",
                                    &h_test_X, &h_test_Y, &N_test);

    printf("Loaded %d training samples", N_train);
    if (have_test)
        printf(", %d test samples", N_test);
    printf("\nArchitecture: %d → %d → %d → %d\n\n", IN, H1, H2, OUT);

    dim3 BLK(16, 16);
    auto G = [](int R, int C)
    { return dim3((R + 15) / 16, (C + 15) / 16); };

    float *W1, *B1, *W2, *B2, *W3, *B3;
    CUDA_CHECK(cudaMalloc(&W1, IN * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B1, H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W2, H1 * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B2, H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W3, H2 * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B3, OUT * sizeof(float)));

    float *X, *Y;
    float *A1, *A2, *Z3, *A3;
    float *loss_vec;
    CUDA_CHECK(cudaMalloc(&X, BATCH * IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y, BATCH * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A1, BATCH * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A2, BATCH * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z3, BATCH * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A3, BATCH * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&loss_vec, BATCH * sizeof(float)));

    float *dZ3, *dA2, *dW3, *dB3;
    float *dA1, *dW2, *dB2;
    float *dW1, *dB1;
    CUDA_CHECK(cudaMalloc(&dZ3, BATCH * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA2, BATCH * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW3, H2 * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB3, OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA1, BATCH * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, H1 * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB2, H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, IN * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB1, H1 * sizeof(float)));

    init_weights<<<G(IN, H1), BLK>>>(IN, H1, W1, 42);
    CUDA_CHECK(cudaMemset(B1, 0, H1 * sizeof(float)));
    init_weights<<<G(H1, H2), BLK>>>(H1, H2, W2, 43);
    CUDA_CHECK(cudaMemset(B2, 0, H2 * sizeof(float)));
    init_weights<<<G(H2, OUT), BLK>>>(H2, OUT, W3, 44);
    CUDA_CHECK(cudaMemset(B3, 0, OUT * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_X_batch_train = new float[BATCH * IN];
    float *h_X_batch_test = new float[BATCH * IN];
    float *h_Y_batch_train = new float[BATCH * OUT];
    float *h_Y_batch_test = new float[BATCH * OUT];
    float *h_loss_buf = new float[BATCH];
    float *h_prebuf = new float[BATCH * OUT];

    int *train_idx = new int[N_train];
    for (int i = 0; i < N_train; i++)
        train_idx[i] = i;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle(train_idx, N_train);

        float epoch_loss = 0.f;
        int n_batches = N_train / BATCH;

        for (int bi = 0; bi < n_batches; bi++)
        {

            for (int i = 0; i < BATCH; i++)
            {
                int s = train_idx[bi * BATCH + i];
                memcpy(h_X_batch_train + i * IN, h_train_X + s * IN, IN * sizeof(float));
                memcpy(h_Y_batch_train + i * OUT, h_train_Y + s * OUT, OUT * sizeof(float));
            }
            CUDA_CHECK(cudaMemcpy(X, h_X_batch_train, BATCH * IN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(Y, h_Y_batch_train, BATCH * OUT * sizeof(float), cudaMemcpyHostToDevice));

            FeedforwardReLU<<<G(BATCH, H1), BLK>>>(BATCH, IN, H1, X, W1, B1, A1);

            FeedforwardReLU<<<G(BATCH, H2), BLK>>>(BATCH, H1, H2, A1, W2, B2, A2);

            Feedforward<<<G(BATCH, OUT), BLK>>>(BATCH, H2, OUT, A2, W3, B3, Z3);
            Softmax<<<(BATCH + 255) / 256, 256>>>(BATCH, OUT, Z3, A3);

            crossEntropyLoss<<<(BATCH + 255) / 256, 256>>>(BATCH, OUT, A3, Y, loss_vec);
            epoch_loss += mean_loss(loss_vec, BATCH, h_loss_buf);

            crossEntropyBackwards<<<G(BATCH, OUT), BLK>>>(BATCH, OUT, A3, Y, dZ3);

            feedforwarlinear_backward<<<G(H2, OUT), BLK>>>(
                BATCH, H2, OUT, A2, dZ3, dW3, dB3);

            feedforwarlinear_backwardx<<<G(BATCH, H2), BLK>>>(
                BATCH, H2, OUT, W3, dZ3, dA2);

            feedforwarrelu_backward<<<G(H1, H2), BLK>>>(
                BATCH, H1, H2, A1, A2, dA2, dW2, dB2);

            feedforwarrelu_backwardx<<<G(BATCH, H1), BLK>>>(
                BATCH, H1, H2, A2, W2, dA2, dA1);

            feedforwarrelu_backward<<<G(IN, H1), BLK>>>(
                BATCH, IN, H1, X, A1, dA1, dW1, dB1);

            update_layer<<<G(H2, OUT), BLK>>>(H2, OUT, BATCH, LR, W3, B3, dW3, dB3);
            update_layer<<<G(H1, H2), BLK>>>(H1, H2, BATCH, LR, W2, B2, dW2, dB2);
            update_layer<<<G(IN, H1), BLK>>>(IN, H1, BATCH, LR, W1, B1, dW1, dB1);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        epoch_loss /= n_batches;

        float test_acc = -1.f;
        if (have_test)
        {
            int correct = 0;
            int n_test_batches = N_test / BATCH;
            for (int bi = 0; bi < n_test_batches; bi++)
            {
                for (int i = 0; i < BATCH; i++)
                {
                    int s = bi * BATCH + i;
                    memcpy(h_X_batch_test + i * IN, h_test_X + s * IN, IN * sizeof(float));
                    memcpy(h_Y_batch_test + i * OUT, h_test_Y + s * OUT, OUT * sizeof(float));
                }
                CUDA_CHECK(cudaMemcpy(X, h_X_batch_test, BATCH * IN * sizeof(float), cudaMemcpyHostToDevice));

                FeedforwardReLU<<<G(BATCH, H1), BLK>>>(BATCH, IN, H1, X, W1, B1, A1);
                FeedforwardReLU<<<G(BATCH, H2), BLK>>>(BATCH, H1, H2, A1, W2, B2, A2);
                Feedforward<<<G(BATCH, OUT), BLK>>>(BATCH, H2, OUT, A2, W3, B3, Z3);
                Softmax<<<(BATCH + 255) / 256, 256>>>(BATCH, OUT, Z3, A3);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(h_prebuf, A3, BATCH * OUT * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < BATCH; i++)
                {
                    if (argmax(h_prebuf + i * OUT, OUT) ==
                        argmax(h_Y_batch_test + i * OUT, OUT))
                        correct++;
                }
            }
            test_acc = (float)correct / (n_test_batches * BATCH) * 100.f;
            printf("Epoch %2d/%d | loss: %.4f | test acc: %.2f%%\n",
                   epoch + 1, EPOCHS, epoch_loss, test_acc);
        }
        else
        {
            printf("Epoch %2d/%d | loss: %.4f\n", epoch + 1, EPOCHS, epoch_loss);
        }
    }

    cudaFree(W1);
    cudaFree(B1);
    cudaFree(W2);
    cudaFree(B2);
    cudaFree(W3);
    cudaFree(B3);
    cudaFree(X);
    cudaFree(Y);
    cudaFree(A1);
    cudaFree(A2);
    cudaFree(Z3);
    cudaFree(A3);
    cudaFree(loss_vec);
    cudaFree(dZ3);
    cudaFree(dA2);
    cudaFree(dW3);
    cudaFree(dB3);
    cudaFree(dA1);
    cudaFree(dW2);
    cudaFree(dB2);
    cudaFree(dW1);
    cudaFree(dB1);

    delete[] h_X_batch_train;
    delete[] h_Y_batch_train;
    delete[] h_X_batch_test;
    delete[] h_Y_batch_test;
    delete[] h_loss_buf;
    delete[] h_prebuf;
    delete[] train_idx;
    free(h_train_X);
    free(h_train_Y);
    if (have_test)
    {
        free(h_test_X);
        free(h_test_Y);
    }

    return 0;
}
