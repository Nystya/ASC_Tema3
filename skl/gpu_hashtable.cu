#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void init_hashtable(Node *table, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		table[idx].key = 0;
		table[idx].value = 0;
	}
}

// __global__ insert_value(Node *table, int size, int *keys, int *value) {
// 	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	int idx;

// 	if (key_idx < size)
// 		idx = hash1(key_idx);

// 	// atomicCAS(table->[idx].key, )
// }

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	cudaSetDevice(0);

	cudaMallocManaged(&this->table, size * sizeof(Node));
	
	init_hashtable <<<size / BLOCKSIZE, BLOCKSIZE>>>(this->table, size);

	cudaDeviceSynchronize();

	this->limit = size;
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->table);
	this->table = NULL;

}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Node *aux_table;

	cudaMallocManaged(&aux_table, numBucketsReshape * sizeof(Node));

	init_hashtable <<<numBucketsReshape / BLOCKSIZE, BLOCKSIZE>>>(this->table, numBucketsReshape);

	cudaDeviceSynchronize();

	cudaFree(this->table);
	
	this->table = aux_table;
	this->limit = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result;

	cudaMallocManaged(&result, numKeys * sizeof(Node));
	if (!result)
		return NULL;

	return result;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return 0.f; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
