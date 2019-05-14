#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void insert_value(Node *table, int size, int *keys, int *values, int numKeys) {
	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;
	int filled;

	if (key_idx >= numKeys)
		return;
	
	idx = ((long)abs(keys[key_idx]) * PRIME1) % PRIME2 % size;

	/* Search a free slot using linear probing */
	/* Use atomic operations so that there are no race conditions */
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

	/* Move values from old hashtable to new hashtable */
	if (old_table[key_idx].filled) {
		idx = ((long)abs(old_table[key_idx].key) * PRIME1) % PRIME2 % new_size;
		
		/* Search a free slot using linear probing */
		/* Use atomic operations so that there are no race conditions */
		filled = atomicCAS(&new_table[idx].filled, 0, 1);
		while (filled) {
			/* If filled with the same key */
			if (new_table[idx].key == old_table[key_idx].key) {
				new_table[idx].value = old_table[key_idx].value;
				return;
			}

			idx = (idx + 1) % new_size;
			filled = atomicCAS(&new_table[idx].filled, 0, 1);
		}

		new_table[idx].key = old_table[key_idx].key;
		new_table[idx].value = old_table[key_idx].value;
	}
}

__global__ void get_node(Node *table, int size, int *keys, int *result, int numKeys) {
	int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx;

	if (key_idx >= numKeys)
		return;
	
	idx = ((long)abs(keys[key_idx]) * PRIME1) % PRIME2 % size;

	/* Search value for key using linear probing */
	while (table[idx].key != keys[key_idx]) {
		idx = (idx + 1) % size;
	}

	result[key_idx] = table[idx].value;
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	int blocks = size / BLOCKSIZE;

	if (size % BLOCKSIZE)
		blocks++;

	this->table = NULL;

	cudaMalloc(&this->table, size * sizeof(Node));
	if (!this->table) {
		printf("[INIT][COULD NOT ALLOC MEMORY]\n");
		exit(-1);
	}
	
	cudaMemset(this->table, 0, size * sizeof(Node));

	this->limit = size;
	this->load = 0;
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

	if (numBucketsReshape % BLOCKSIZE)
		blocks++;

	/* I need a new table */
	cudaMallocManaged(&new_table, numBucketsReshape * sizeof(Node));
	if (!new_table) {
		printf("[RESHAPE][COULD NOT ALLOC MEMORY]\n");
		exit(-1);
	}

	/* Init the new table */
	cudaMemset(new_table, 0, numBucketsReshape * sizeof(Node));
	cudaDeviceSynchronize();

	/* Move values from old hashtable to the new one */
	reshape_map <<<blocks, BLOCKSIZE>>> (this->table, new_table, this->limit, numBucketsReshape);

	cudaDeviceSynchronize();
	if (cudaSuccess != cudaGetLastError()) {
		printf("Reshape failed\n");
		exit(-1);
	}

	/* Free memory for old hashtable */
	cudaFree(this->table);

	this->table = new_table;
	this->limit = numBucketsReshape;
}

/* Helper function */
float checkLoadFactor(float load, float limit) {
	return load / limit;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int blocks = numKeys / BLOCKSIZE;
	int *gpukeys, *gpuvalues;

	if (numKeys % BLOCKSIZE)
		blocks++;

	/* Check load factor to see if it is needed to reshape */
	if (checkLoadFactor((float)(this->load + numKeys), this->limit) >= 0.75) {
		reshape(1.5 * this->limit);
	}

	/* Keys sent to kernel must be in VRAM */
	cudaMalloc(&gpukeys, numKeys * sizeof(int));
	if (!gpukeys) {
		printf("GPU keys fail\n");
	}

	/* Values sent to kernel must be in VRAM */
	cudaMalloc(&gpuvalues, numKeys * sizeof(int));
	if (!gpukeys) {
		printf("GPU values fail\n");
	}

	/* Move keys and values from RAM to VRAM */
	cudaMemcpy(gpukeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuvalues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* Compute index and store data in hashtable */
	insert_value <<<blocks, BLOCKSIZE>>>(this->table, this->limit, gpukeys, gpuvalues, numKeys);
	cudaDeviceSynchronize();

	cudaFree(gpukeys);
	cudaFree(gpuvalues);

	this->load += numKeys;

	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result ;
	int *gpuresult;
	int *gpukeys;
	int blocks = numKeys / BLOCKSIZE;

	if (numKeys % BLOCKSIZE)
		blocks++;

	/* Result has to be in RAM */
	result = (int *) malloc (numKeys * sizeof(int));
	if (!result) {
		printf("[GET1][COULD NOT ALLOC MEMORY]\n");
		exit(-1);
	}
	
	/* Keys sent to kernel must be in VRAM */
	cudaMalloc(&gpukeys, numKeys * sizeof(int));
	if (!gpukeys) {
		printf("GPU keys fail\n");
	}

	/* Move keys from host to device */
	cudaMemcpy(gpukeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* Kernel operations must be done in VRAM */
	cudaMalloc(&gpuresult, numKeys * sizeof(int));
	if (!gpuresult) {
		printf("[GET2][COULD NOT ALLOC MEMORY]\n");
		exit(-1);
	}
	
	/* Find results in hashtable */
	get_node <<<blocks, BLOCKSIZE>>> (this->table, this->limit, gpukeys, gpuresult, numKeys);

	cudaDeviceSynchronize();

	/* Move results from VRAM to RAM */
	cudaMemcpy(result, gpuresult, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(gpukeys);
	cudaFree(gpuresult);

	return result;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return (float)this->load / this->limit;
}


/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
