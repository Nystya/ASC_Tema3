#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	// cudaMallocManaged(&this->table, size * sizeof(Node));
	// if (!this->table)
		// return NULL;

	int i;

	this->table = (Node *) malloc(size * sizeof(Node));
	
	for (i = 0; i < size; i++) {
		this->table[i].key = NULL;
		this->table[i].value = NULL;
	}

	this->limit = size;
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	free(this->table);
	this->table = NULL;

}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int i;
	int idx;
	int auxidx;

	cout << "Adding " << numKeys << " items.\n";

	for (i = 0; i < numKeys; i++) {
		idx = hash1(keys[i], this->limit);
		if (this->table[idx].value == NULL || *(this->table[idx].key) == keys[i]) {
			this->table[idx].key = (int *) malloc(sizeof(int));
			this->table[idx].value = (int *) malloc(sizeof(int));
			memcpy(this->table[idx].key, &keys[i], sizeof(int));
			memcpy(this->table[idx].value, &values[i], sizeof(int));

			cout << "Adding " << values[i] << " on position " << idx << " for key " << keys[i] << "\n";
		} else {
			auxidx = idx;
			idx = (idx + 1) % this->limit;

			while (auxidx != idx && this->table[idx % this->limit].value != NULL)
				idx = (idx + 1) % this->limit;
			
			if (auxidx == idx) {
				reshape(this->limit);
				i--;
				continue;
			}

			this->table[idx].key = (int *) malloc(sizeof(int));
			this->table[idx].value = (int *) malloc(sizeof(int));
			memcpy(this->table[idx].key, &keys[i], sizeof(int));
			memcpy(this->table[idx].value, &values[i], sizeof(int));

			cout << "Collision: Adding " << values[i] << " on position " << idx << " for key " << keys[i] << "\n";
		}
	}

	cout << "Done adding items\n";

	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int i;
	int idx;
	int auxidx;
	int *result = (int *) malloc(numKeys * sizeof(int));

	if (!result)
		return NULL;

	for (i = 0; i < numKeys; i++) {
		idx = hash1(keys[i], this->limit);
		
		if (*(this->table[idx].key) == keys[i]) {
			result[i] = *(this->table[idx].value);
		} else {
			auxidx = idx;
			idx = (idx + 1) % this->limit;

			while ( auxidx != idx && *(this->table[idx % this->limit].key) != keys[i]) {
				idx = (idx + 1) % this->limit;
			}

			if (auxidx == idx) {
				return NULL;
			}

			result[i] = *(this->table[idx].value);
		}
	}

	return result;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return 0.f; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(100);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
