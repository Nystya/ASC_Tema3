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
		table[idx].filled = 0;
	}
}

__global__ void insert_value(Node *table, int size, int *keys, int *result, int numKeys) {
	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;
	int filled;

	if (key_idx >= numKeys)
		return;
	
	idx = hash1(keys[key_idx], size);

	/* Search a free slot using linear probing */
	filled = atomicCAS(&table[idx].filled, 0, 1);
	while (filled) {
		/* If filled with the same key */
		if (table[idx].key == keys[key_idx]) {
			table[idx].value = values[key_idx];
			return;
		}

		idx = (idx + 1) % size;
		filled = atomicCAS(&table[idx].filled, 0, 1);
	}

	table[idx].key = keys[key_idx];
	table[idx].value = values[key_idx];
}

__global__ void reshape_map(Node *old_table, Node *new_table, int old_size, int new_size) {
	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;
	int filled;

	if (key_idx >= old_size || key_idx >= new_size)
		return;
	
	if (old_table[key_idx].filled) {
		idx = hash1(old_table[key_idx].key, size);

		filled = atomicCAS(&new_table[idx].filled, 0, 1);
		while (filled) {
			/* If filled with the same key */
			if (new_table[idx].key == old_table[key_idx].key) {
				new_table[idx].value = old_table[key_idx].value;
				return;
			}

			idx = (idx + 1) % size;
			filled = atomicCAS(&new_table[idx].filled, 0, 1);
		}

		new_table[idx].key = old_table[key_idx].key;
		new_table[idx].value = old_table[key_idx].value;
	}
}

__global__ void get_node(Node *table, int size, int *keys, int *values, int numKeys) {
	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;
	int filled;

	if (key_idx >= numKeys)
		return;
	
	idx = hash1(keys[key_idx], size);

	/* Search a free slot using linear probing */
	while (table[idx].key != keys[key_idx]) {
		idx = (idx + 1) % size;
	}

	result[key_idx] = table[idx].value;
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	int blocks = numKeys / BLOCKSIZE;

	if (numKeys % BLOCKSIZE)
		blocks++;

	cudaSetDevice(0);

	cudaMallocManaged(&this->table, size * sizeof(Node));
	
	init_hashtable <<<blocks, BLOCKSIZE>>>(this->table, size);

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
	Node *new_table;
	int blocks = numBucketsReshape / BLOCKSIZE;

	if (numKeys % BLOCKSIZE)
		blocks++;

	cudaMallocManaged(&new_table, numBucketsReshape * sizeof(Node));

	init_hashtable <<<blocks, BLOCKSIZE>>> (this->table, numBucketsReshape);

	cudaDeviceSynchronize();

	reshape_map <<<blocks, BLOCKSIZE>>> (this->table, new_table, this->limit, numBucketsReshape);

	cudaDeviceSynchronize();

	cudaFree(this->table);
	
	this->table = aux_table;
	this->limit = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int blocks = numKeys / BLOCKSIZE;

	if (numKeys % BLOCKSIZE)
		blocks++;

	if (checkLoadFactor((float)(this->load + numKeys), this->limit) >= 0.75)
		reshape(2 * this->limit);

	insert_value <<<blocks, BLOCKSIZE>>>(this->table, this->limit, keys, values, numKeys);

	cudaDeviceSynchronize();
	
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result;
	int blocks = numKeys / BLOCKSIZE;

	if (numKeys % BLOCKSIZE)
		blocks++;

	cudaMallocManaged(&result, numKeys * sizeof(Node));
	if (!result)
		return NULL;

	get_node <<<blocks, BLOCKSIZE>>> (this->table, this->limit, keys, result, numKeys);

	cudaDeviceSynchronize();
	
	return result;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return this->load / this->limit;
}

float checkLoadFactor(float load, float limit) {
	return load / limit;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
